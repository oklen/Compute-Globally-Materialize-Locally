"""rA_graph.py -- E1's graph arm: does a DIRECTED entity-relation index rank value rows above
mention-only rows, where every text-index layer fails?

WHY THIS ARM IS THE ONE THAT MATTERS. The four text arms are done and they agree: on multi-hop,
BM25 is significantly WORSE than a random permutation (delta +.264), dense repairs that to chance
(-.024), and the full deployed stack with a cross-encoder stops at chance (-.008, CI crosses 0) --
while the SAME components take single-hop from -.197 to -.496. So the ceiling is not weak
components; it is what a text index can express. A directed graph can express the one distinction
the text index cannot: "this entity is the SUBJECT of a value assertion" vs "this entity is merely
mentioned". That is exactly the M/V boundary. Zep/Graphiti and Mem0-Graph both ship this layer.

  goes below chance on multi-hop  -> the field already has the fix, and we can say what it fixes
  stays at chance                 -> a strong negative about the graph layer as deployed

RANKING FROM A GRAPH IS NOT THE SAME OBJECT AS RANKING FROM AN INDEX, and pretending otherwise
would fake the comparison. `Graphiti.search` returns EntityEdges, not turns, and it returns the top
`num_results` of them, not a total order over the pool. We therefore:
  * ingest ONE TURN PER EPISODE with a deterministic uuid, so every edge carries turn provenance;
  * score a turn by the best rank of any returned edge citing it;
  * place every turn NO edge cites below all cited turns, ordered by the same recency tiebreak the
    other arms use -- and REPORT the coverage (how many of V and M were cited at all), because a
    pi computed mostly out of the uncited tail is a statement about the tiebreak, not the graph.

COST. Graphiti runs several LLM calls per episode (entity extraction, edge extraction, dedup). At
~5,882 turns that is tens of thousands of calls, so this module ingests one conversation at a time,
checkpoints the Kuzu database to disk, and reports calls/turn from the pilot before the rest is run.

    python -m kvmemory.rA_graph ingest --data ./data/locomo10.json --conv 0 --db ./out/rA_graph
    python -m kvmemory.rA_graph rank   --data ./data/locomo10.json --labels ./out/rA_labels.json --conv 0 \
        --db ./out/rA_graph --out ./out/rA_graph_conv0.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# OpenAI-compatible LLM endpoint for graphiti's extraction and reranking (no defaults; the reported
# run used the re-judge's chat model with reasoning disabled). See kvmemory/llm_api.py.
LLM_BASE = os.environ.get("JUDGE_BASE_URL", "")
LLM_KEY = os.environ.get("JUDGE_API_KEY", "")
LLM_MODEL = os.environ.get("JUDGE_MODEL", "")
EMB_PATH = os.environ.get("RA_EMB_PATH", "./models/Qwen3-Embedding-0.6B")


# ---------------------------------------------------------------- embedder

def _EmbedderBase():
    """graphiti validates the embedder with `isinstance(x, EmbedderClient)` through pydantic, so a
    duck-typed object is rejected; we must actually subclass it. Imported lazily so this module can
    be inspected without graphiti installed."""
    from graphiti_core.embedder.client import EmbedderClient
    return EmbedderClient


class LocalEmbedder(_EmbedderBase()):
    """graphiti EmbedderClient over the SAME Qwen3-Embedding-0.6B the dense arm uses, so the graph
    arm's vector half is not a different model from the arm it is being compared against.

    Right padding + gather at the last real token: left padding plus last-token pooling returns NaN
    on some backends and NaN does not raise -- see rA_arms.DenseArm."""

    def __init__(self, path=EMB_PATH, device="cpu", batch=16, max_len=512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tok.padding_side = "right"
        self.model = AutoModel.from_pretrained(
            path, dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=False
        ).to(device).eval()
        self.device, self.batch, self.max_len = device, batch, max_len

    def _enc(self, texts):
        import numpy as np
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch):
                e = self.tok(texts[i:i + self.batch], return_tensors="pt", padding=True,
                             truncation=True, max_length=self.max_len).to(self.device)
                H = self.model(**e).last_hidden_state
                last = e["attention_mask"].sum(1) - 1
                h = H[self.torch.arange(H.size(0), device=H.device), last]
                h = self.torch.nn.functional.normalize(h.float(), p=2, dim=-1)
                out.append(h.cpu().numpy())
        E = np.concatenate(out, 0)
        assert np.isfinite(E).all(), "embedder returned NaN/inf"
        return E

    async def create(self, input_data):
        if isinstance(input_data, str):
            return self._enc([input_data])[0].tolist()
        return self._enc(list(input_data))[0].tolist()

    async def create_batch(self, input_data_list):
        return [v.tolist() for v in self._enc(list(input_data_list))]


# ---------------------------------------------------------------- graph build

_LLM_CALLS = [0]


def _thinking_off(client):
    """The reported endpoint ran a reasoning pass by default: the request sits for 80-90s and comes
    back with the reasoning trace instead of the answer, which surfaces to the caller as a read
    timeout.  Measured on this endpoint: 90.1s timeout with the default, 1.4s with the switch off.
    The OpenAI-compatible surface takes the switch in extra_body."""
    if client is None or not hasattr(client, "chat"):
        return client
    orig = client.chat.completions.create

    async def create(**kw):
        eb = dict(kw.pop("extra_body", None) or {})
        eb.setdefault("thinking", {"type": "disabled"})
        _LLM_CALLS[0] += 1
        return await orig(extra_body=eb, **kw)

    client.chat.completions.create = create
    return client


KUZU_BUFFER_BYTES = int(os.environ.get("KUZU_BUFFER_BYTES", 1536 * 1024 * 1024))
KUZU_THREADS = int(os.environ.get("KUZU_THREADS", 4))


def _cap_kuzu_resources():
    """KuzuDriver constructs `kuzu.Database(db)` with no resource arguments, so Kuzu falls back to
    its own auto-sizing: buffer pool ~= 80% of SYSTEM memory and threads = every core.  Inside a
    container both readings are wrong -- it sees the host's 503 GB and 123 cores while the cgroup
    caps us at 32 GB.  Four ingests then grow their buffer pools until the cgroup OOM-kills them,
    and an OOM kill leaves no traceback: the log simply stops mid-run (cgroup oom_kill counter is
    the only evidence).  Cap both.  This changes no query semantics, only how much memory and how
    many threads one embedded database is allowed to take."""
    import kuzu
    if getattr(kuzu.Database, "_rA_capped", False):
        return
    real = kuzu.Database

    def capped(path=None, **kw):
        kw.setdefault("buffer_pool_size", KUZU_BUFFER_BYTES)
        kw.setdefault("max_num_threads", KUZU_THREADS)
        return real(path, **kw)

    capped._rA_capped = True
    kuzu.Database = capped


def _create_fts_indices(drv):
    """graphiti 0.11.3 DEFINES the Kuzu full-text indices (graph_queries.get_fulltext_indices) and
    then never runs them: KuzuDriver.build_indices_and_constraints() is a no-op whose comment says
    'indices are created during setup_schema()', but SCHEMA_QUERIES only creates tables.  Every
    search path meanwhile emits CALL QUERY_FTS_INDEX(...), so the first retrieval dies with
    `Binder exception: Table RelatesToNode_ doesn't have an index with name edge_name_and_fact`.

    We therefore run the library's OWN index statements -- not a reimplementation and not a change
    to any ranking behaviour, just the call the Kuzu driver forgot to make.  Verified separately:
    the same query binder-errors before these run, returns scored rows after, and picks up rows
    inserted LATER, so creating the indices at open time is enough and no post-ingest rebuild is
    needed."""
    import kuzu
    from graphiti_core.graph_queries import get_fulltext_indices
    from graphiti_core.driver.driver import GraphProvider

    conn = kuzu.Connection(drv.db)
    stmts = get_fulltext_indices(GraphProvider.KUZU)
    made, skipped = [], []
    for q in stmts:
        name = q.split("'")[3]
        try:
            conn.execute(q)
            made.append(name)
        except Exception as e:
            skipped.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
    # Verify rather than assume: a swallowed "already exists" and a genuinely missing index look
    # identical from the outside, and a missing one does not fail loudly -- it fails 300 turns later
    # inside add_episode's edge dedup, after the graph has been built wrong.
    probe = ("CALL QUERY_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', 'probe', TOP := 1) "
             "RETURN node.uuid")
    try:
        conn.execute(probe)
    except Exception as e:
        raise RuntimeError(
            f"Kuzu full-text index missing after setup ({len(stmts)} statements, "
            f"created={made}, skipped={skipped}): {e}") from e
    conn.close()
    return made


def _client_and_graph(db_path, group):
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

    cfg = LLMConfig(api_key=LLM_KEY, base_url=LLM_BASE, model=LLM_MODEL, small_model=LLM_MODEL)
    llm = OpenAIGenericClient(config=cfg)
    emb = LocalEmbedder()
    xenc = OpenAIRerankerClient(config=cfg)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    _cap_kuzu_resources()
    drv = KuzuDriver(db=db_path)
    # graphiti 0.11.3's base GraphDriver declares `_database: str` and generic code reads it, but
    # KuzuDriver never assigns it -- every add_episode dies with AttributeError. The Kuzu backend is
    # marked deprecated upstream, which is presumably why nobody hit this. Kuzu is a single-file
    # embedded store with no server, which is exactly what we want for a reproducible arm, so we set
    # the attribute the base class promises rather than standing up a Neo4j.
    if not hasattr(drv, "_database"):
        drv._database = "kuzu"
    made = _create_fts_indices(drv)
    print(f"[graph] fts indices ready: {made}", flush=True)
    _thinking_off(getattr(llm, "client", None))
    _thinking_off(getattr(xenc, "client", None))
    return Graphiti(graph_driver=drv, llm_client=llm, embedder=emb, cross_encoder=xenc)


def map_path(db_path):
    """turn -> episode uuid, written beside the database.

    We do NOT pass our own uuid to add_episode: in graphiti 0.11.3 that argument means "update the
    episode that already has this uuid" and raises NodeNotFoundError when it does not exist. So the
    library assigns the uuid and we persist the mapping, which is what turn-level provenance -- and
    therefore the whole arm -- depends on."""
    return os.path.splitext(db_path)[0] + ".turnmap.json"


async def ingest(args):
    from kvmemory.locomo import load_locomo
    from kvmemory.rA_roles import _utterance
    eps = load_locomo(args.data)
    ep = next(e for e in eps if str(e.conv_id) == str(args.conv))
    segs = ep.segments[args.start:args.limit or None]
    g = _client_and_graph(args.db, args.conv)
    await g.build_indices_and_constraints()

    mp = map_path(args.db)
    tmap = json.load(open(mp)) if os.path.exists(mp) else {}
    t0, done = time.time(), 0
    miss = []
    import kuzu as _kz

    def _checkpoint():
        """Fold the WAL into the database file periodically.  A SIGKILLed Kuzu database is not
        merely unclean -- reopening one and continuing to insert put the process into a rewrite
        loop (117 MB/s of pwrite against a 24 MB file, zero turns completed, reproduced on two
        separate databases).  Checkpointing keeps the on-disk file close to current so that at
        worst little is lost; the launcher rebuilds rather than resumes regardless."""
        try:
            c = _kz.Connection(g.driver.db)
            c.execute("CHECKPOINT")
            c.close()
        except Exception as e:
            print(f"  checkpoint skipped: {type(e).__name__}: {str(e)[:80]}", flush=True)
    prev = None
    for s in segs:
        ts = (s.meta or {}).get("timestamp") or ""
        try:
            when = datetime.strptime(ts, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
        except Exception:
            when = datetime(2023, 1, 1, tzinfo=timezone.utc)
        speaker = (s.meta or {}).get("speaker") or "speaker"
        if str(s.turn) in tmap:                        # resume: already ingested
            prev = tmap[str(s.turn)]
            done += 1
            continue
        t1 = time.time()
        # RETRY, do not skip.  The first sweep dropped 99 turns across ten conversations, every
        # one of them a RateLimitError, and the loop merely logged and moved on.  A turn that
        # never became an episode can never be cited by an edge, so it falls into the uncited tail
        # of the ranking -- a silent, one-directional bias in exactly the statistic this arm
        # reports.  Losing throughput to backoff is the cheap side of that trade.
        ok = False
        for attempt in range(6):
            try:
                res = await g.add_episode(
                    name=f"turn{s.turn}", episode_body=f"{speaker}: {_utterance(s)}",
                    source_description=f"LoCoMo conversation {args.conv}", reference_time=when,
                    group_id=f"conv{args.conv}",
                    previous_episode_uuids=[prev] if prev else None)
                tmap[str(s.turn)] = res.episode.uuid
                prev = res.episode.uuid
                ok = True
                break
            except Exception as e:
                name = type(e).__name__
                if attempt == 5:
                    print(f"  turn {s.turn} FAILED {name}: {str(e)[:160]}", flush=True)
                    break
                await asyncio.sleep(min(60.0, 2.0 * (2.5 ** attempt)) * (0.5 + random.random()))
        if not ok:
            miss.append(s)
        done += 1
        # flush EVERY turn, not every ten: a kill between flushes leaves those episodes committed
        # in Kuzu but missing from the map, and the resume ingests them a second time -- silent
        # duplicate episodes for the same turn, which is exactly the provenance the arm rests on.
        json.dump(tmap, open(mp, "w"))
        if done % 50 == 0:
            _checkpoint()
        el = time.time() - t0
        if done <= 5 or done % 10 == 0:
            print(f"  {done}/{len(segs)} turns | last {time.time()-t1:.1f}s | {el/done:.1f}s/turn "
                  f"| {_LLM_CALLS[0]/max(1,done):.1f} llm-calls/turn "
                  f"| eta {(len(segs)-done)*el/done/60:.0f} min", flush=True)
    # repair pass: give every turn that exhausted its retries one more chance at the end, when the
    # burst that rate-limited it is over.  Reported either way -- a graph missing turns is not the
    # graph, and the reader has to be told how many.
    if miss:
        print(f"  repair pass over {len(miss)} turns that exhausted their retries", flush=True)
        still = []
        for s2 in miss:
            ts2 = (s2.meta or {}).get("timestamp") or ""
            try:
                when2 = datetime.strptime(ts2, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
            except Exception:
                when2 = datetime(2023, 1, 1, tzinfo=timezone.utc)
            sp2 = (s2.meta or {}).get("speaker") or "speaker"
            done_it = False
            for attempt in range(6):
                try:
                    r2 = await g.add_episode(
                        name=f"turn{s2.turn}", episode_body=f"{sp2}: {_utterance(s2)}",
                        source_description=f"LoCoMo conversation {args.conv}",
                        reference_time=when2, group_id=f"conv{args.conv}")
                    tmap[str(s2.turn)] = r2.episode.uuid
                    done_it = True
                    break
                except Exception:
                    await asyncio.sleep(min(90.0, 3.0 * (2.5 ** attempt)) * (0.5 + random.random()))
            if not done_it:
                still.append(s2.turn)
        print(f"  repaired {len(miss) - len(still)}/{len(miss)}; still missing: {still}", flush=True)
    json.dump(tmap, open(mp, "w"))
    _checkpoint()
    print(f"INGEST MISSING conv={args.conv} n={len(segs) - len(tmap)}", flush=True)
    print(f"INGEST DONE conv={args.conv} turns={done} mapped={len(tmap)} "
          f"wall={(time.time()-t0)/60:.1f} min", flush=True)
    await g.close()


# ---------------------------------------------------------------- ranking

async def rank(args):
    from kvmemory.locomo import load_locomo
    eps = load_locomo(args.data)
    ep = next(e for e in eps if str(e.conv_id) == str(args.conv))
    turns = [s.turn for s in ep.segments]
    idx_of_turn = {t: i for i, t in enumerate(turns)}
    tmap = json.load(open(map_path(args.db)))
    uuid_of = {u: int(t) for t, u in tmap.items()}

    recs = [r for r in json.load(open(args.labels))["records"]
            if str(r["conv"]) == str(args.conv) and r["nV"]["N1"] > 0 and r["nM"] > 0]
    g = _client_and_graph(args.db, args.conv)

    out = []
    for r in recs:
        edges = await g.search(r["question"], group_ids=[f"conv{args.conv}"],
                               num_results=args.num_results)
        best = {}                       # turn index -> best (smallest) edge rank citing it
        for rank_i, e in enumerate(edges):
            for eu in (getattr(e, "episodes", None) or []):
                t = uuid_of.get(eu)
                if t is None:
                    continue
                i = idx_of_turn[t]
                if i not in best:
                    best[i] = rank_i
        cited = sorted(best, key=lambda i: (best[i], -turns[i]))
        rest = sorted((i for i in range(len(turns)) if i not in best),
                      key=lambda i: -turns[i])
        order = cited + rest
        pos = [0] * len(turns)
        for p, i in enumerate(order):
            pos[i] = p
        V, M = r["V"]["N1"], r["M_strict"]
        rv, rm = min(pos[i] for i in V), min(pos[i] for i in M)
        out.append({
            "conv": args.conv, "qi": r["qi"], "cat": r["cat"],
            "nV": len(V), "nM": len(M), "chance": len(M) / (len(M) + len(V)),
            "rank_V": rv, "rank_M": rm, "beat": int(rm < rv),
            "n_edges": len(edges), "n_cited_turns": len(best),
            # coverage: a pi decided inside the uncited tail is a statement about the recency
            # tiebreak, not about the graph. Reported per item, aggregated below.
            "V_cited": sum(1 for i in V if i in best), "M_cited": sum(1 for i in M if i in best),
            "decided_in_tail": int(not any(i in best for i in V) and not any(i in best for i in M)),
        })
        if len(out) % 20 == 0:
            print(f"  ranked {len(out)}/{len(recs)}", flush=True)
    await g.close()

    import statistics
    n = len(out)
    if n:
        cov_v = statistics.fmean(x["V_cited"] > 0 for x in out)
        cov_m = statistics.fmean(x["M_cited"] > 0 for x in out)
        tail = statistics.fmean(x["decided_in_tail"] for x in out)
        pi = statistics.fmean(x["beat"] for x in out)
        ch = statistics.fmean(x["chance"] for x in out)
        print(f"conv {args.conv} | n={n} | pi={pi:.3f} chance={ch:.3f} DELTA={pi-ch:+.3f}")
        print(f"  coverage: >=1 V cited on {cov_v:.2f}, >=1 M cited on {cov_m:.2f}, "
              f"decided entirely in the uncited tail on {tail:.2f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"conv": args.conv, "num_results": args.num_results, "items": out},
              open(args.out, "w"), indent=1)
    print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "rank"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--conv", required=True)
        p.add_argument("--db", required=True)
        if name == "ingest":
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--limit", type=int, default=0)
        else:
            p.add_argument("--labels", required=True)
            p.add_argument("--num-results", type=int, default=100)
            p.add_argument("--out", required=True)
    args = ap.parse_args()
    if not LLM_KEY:
        sys.exit("set JUDGE_BASE_URL, JUDGE_API_KEY and JUDGE_MODEL")
    asyncio.run(ingest(args) if args.cmd == "ingest" else rank(args))


if __name__ == "__main__":
    main()
