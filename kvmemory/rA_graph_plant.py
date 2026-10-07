"""Planted-row positive control for the GRAPH arm.

The plan makes this a precondition, not a nicety: without it a weak graph delta cannot be told
apart from a graph that simply does not work, and we would be writing an implementation problem
up as a method problem.  The five text arms have had this gate since the corrected pi run
(bm25/overlap/rrf 200/200, dense 195/200, rerank 191/200 strict); the graph arm had nothing.

Same construction as rA_rank's gate so the numbers are comparable:
    planted body = "<speaker>: <question> <gold> <gold>"
and the same PASS threshold, rank 0 on >= 95% of items.

Two things this is careful about:

* IT NEVER TOUCHES THE REAL GRAPH.  Every conversation's database is copied to local
  scratch first.  Planting writes entity nodes and edges, so planting into ./out/rAg/cN.kz
  would corrupt the very ranking this control exists to validate.
* It removes each planted episode before testing the next one where the installed graphiti
  supports it, and says so in the output when it cannot.  Accumulated plants are not neutral --
  they are extra answer-shaped text in the index -- so which mode ran has to be visible.

    JUDGE_BASE_URL=https://your-endpoint/v1 JUDGE_API_KEY=... JUDGE_MODEL=your-model \\
    JUDGE_EXTRA_BODY='{"thinking": {"type": "disabled"}}' PYTHONPATH=. \\
    RA_EMB_PATH=./models/Qwen3-Embedding-0.6B python -m kvmemory.rA_graph_plant --data ./data/locomo10.json \\
        --conv 0 --db ./out/rA_graph --labels ./out/rA_labels.json --out ./out/rA_graph_plant_conv0.json
"""
import argparse, asyncio, json, os, random, shutil, sys, time


def _self_write_mb():
    """Bytes THIS process has written, from /proc/self/io.

    This is the measurement whose absence once caused a misdiagnosis: free space was falling on a
    shared scratch disk and was attributed to this script, when other jobs were writing to the same
    disk (df said 1.6 TB used, du said 508 MB ours). df cannot attribute a write to a process;
    /proc/self/io can."""
    try:
        for line in open("/proc/self/io"):
            if line.startswith("write_bytes:"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("nan")


def _db_mb(path):
    return sum(os.path.getsize(path + s) for s in ("", ".wal") if os.path.exists(path + s)) / 2 ** 20

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run(args):
    from kvmemory.locomo import load_locomo
    from kvmemory.rA_graph import _client_and_graph, map_path

    eps = load_locomo(args.data)
    ep = next(e for e in eps if str(e.conv_id) == str(args.conv))
    turns = [s.turn for s in ep.segments]
    idx_of_turn = {t: i for i, t in enumerate(turns)}
    tmap = json.load(open(map_path(args.db)))
    uuid_of = {u: int(t) for t, u in tmap.items()}

    recs = [r for r in json.load(open(args.labels))["records"]
            if str(r["conv"]) == str(args.conv) and r["nV"]["N1"] > 0 and r["nM"] > 0]
    random.Random(1234).shuffle(recs)
    recs = recs[:args.n]
    if not recs:
        print("conv %s: no eligible records" % args.conv); return

    # --- work on a copy; planting mutates the graph
    scratch = os.path.join(args.scratch, "plant_c%s.kz" % args.conv)
    os.makedirs(args.scratch, exist_ok=True)
    for suffix in ("", ".wal"):
        src = args.db + suffix
        if os.path.exists(src):
            shutil.copyfile(src, scratch + suffix)
    shutil.copyfile(map_path(args.db), map_path(scratch))
    start_db = _db_mb(scratch)
    print("conv %s: working on a copy at %s (%.1f MB)" % (args.conv, scratch, start_db), flush=True)

    g = _client_and_graph(scratch, args.conv)
    can_remove = hasattr(g, "remove_episode")
    print("  per-item cleanup: %s" % ("remove_episode" if can_remove else
          "UNAVAILABLE -- plants accumulate, which makes this a STRICTLY HARDER test"), flush=True)

    speaker = (ep.segments[0].meta or {}).get("speaker") or "speaker"
    hits, tried, fails = 0, 0, []
    from datetime import datetime, timezone
    t_conv, w_conv = time.time(), _self_write_mb()
    print("  start: self_written=%.1f MB  db=%.1f MB" % (w_conv, _db_mb(scratch)), flush=True)
    for r in recs:
        t_item, w_item = time.time(), _self_write_mb()
        body = "%s: %s %s %s" % (speaker, r["question"], r["gold"], r["gold"])
        try:
            res = await g.add_episode(
                name="plant_q%s" % r["qi"], episode_body=body,
                source_description="planted positive control",
                reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                group_id="conv%s" % args.conv)
            planted_uuid = res.episode.uuid
        except Exception as e:
            print("  qi=%s plant FAILED %s: %s" % (r["qi"], type(e).__name__, str(e)[:120]), flush=True)
            continue
        tried += 1

        edges = await g.search(r["question"], group_ids=["conv%s" % args.conv],
                               num_results=args.num_results)
        # Reproduce rank()'s ordering exactly, with the plant as one more turn.  It is given the
        # newest turn id so that a tie on edge rank breaks its way -- the same benefit the text
        # gate gives it by appending it to the end of the pool.
        PLANT = -1
        best = {}
        for rank_i, e in enumerate(edges):
            for eu in (getattr(e, "episodes", None) or []):
                if eu == planted_uuid:
                    best.setdefault(PLANT, rank_i); continue
                t = uuid_of.get(eu)
                if t is None: continue
                i = idx_of_turn[t]
                best.setdefault(i, rank_i)
        newest = max(turns) + 1
        key = lambda i: (best[i], -(newest if i == PLANT else turns[i]))
        order = sorted(best, key=key)
        at0 = bool(order and order[0] == PLANT)
        hits += at0
        if not at0:
            fails.append({"qi": r["qi"], "plant_edge_rank": best.get(PLANT),
                          "winner_edge_rank": best[order[0]] if order else None,
                          "plant_cited": PLANT in best})
        if can_remove:
            try: await g.remove_episode(planted_uuid)
            except Exception as e: print("  cleanup failed on qi=%s: %s" % (r["qi"], type(e).__name__), flush=True)
        # per-ITEM debug (was every 5th): item wall time, this process's own write volume, and
        # the scratch database size. If any of these grows without bound, it is visible here on
        # the item it starts, not hours later in an aggregate.
        print("  [%2d/%2d] qi=%-4s %-6s edge_rank=%-4s %5.1fs  self_written=%8.1f MB (+%.1f)  db=%6.1f MB"
              % (tried, len(recs), r["qi"], "RANK0" if at0 else "miss",
                 str(best.get(PLANT)), time.time() - t_item,
                 _self_write_mb(), _self_write_mb() - w_item, _db_mb(scratch)), flush=True)
    await g.close()

    print("  conv %s totals: %.0fs wall, %.1f MB written BY THIS PROCESS, db %+.1f MB"
          % (args.conv, time.time() - t_conv, _self_write_mb() - w_conv, _db_mb(scratch) - start_db),
          flush=True)
    rate = hits / tried if tried else 0.0
    verdict = "PASS" if tried and rate >= 0.95 else "FAIL"
    print("conv %s: planted row at rank 0 on %d/%d = %.3f   %s" % (args.conv, hits, tried, rate, verdict))
    if fails:
        never = sum(1 for f in fails if not f["plant_cited"])
        print("  of %d misses, %d had the planted episode cited by NO edge at all "
              "(that is a retrieval failure, not a ranking one)" % (len(fails), never))
    json.dump({"conv": args.conv, "n": tried, "hits": hits, "rate": rate, "verdict": verdict,
               "cleanup": "remove_episode" if can_remove else "accumulating",
               "misses": fails}, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    for suffix in ("", ".wal"):
        try: os.remove(scratch + suffix)
        except OSError: pass
    try: os.remove(map_path(scratch))
    except OSError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-results", type=int, default=100)
    ap.add_argument("--scratch", default="./out/rAg_plant")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
