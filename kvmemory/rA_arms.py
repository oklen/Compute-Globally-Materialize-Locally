"""rA_arms.py -- the retrieval stack for E1, one class per deployed layer.

E1's first round ran only the two cheapest arms (BM25 + an unweighted-overlap negative control) and
found they are statistically indistinguishable: IDF and length normalisation buy no role
discrimination at all. That is expected -- both are functions of the term multiset, and the
synthetic result (paper section 7) proves the term multiset does not separate a row that carries
a value from a row that points at it. It says NOTHING about the layers real systems actually run.

The deployed stack, in the order a survey of deployed agent-memory stacks found it:

    lexical  ->  dense  ->  rank fusion  ->  cross-encoder rerank  ->  directed graph

Every layer here is the library/implementation kvmemory already uses, not a re-implementation:
  DenseArm    mirrors components.EmbeddingRouter (Qwen3-Embedding-0.6B, last-token pool, L2, cosine,
              the Instruct prefix on QUERIES only -- the Qwen3-Embedding convention)
  RRFArm      mirrors components.HybridRouter (reciprocal rank fusion, c=60, recency tiebreak)
  RerankArm   FlagEmbedding's bge-reranker-v2-m3 applied to the top-`depth` of a base arm, which is
              how a cross-encoder is actually deployed -- it never scores the whole corpus. Rows
              below `depth` keep their base order underneath the reranked head, so a full ranking
              still exists and min-rank over M/V is always defined.

WHY THE INTERFACE CHANGED. Scoring used to be fn(query, token_lists). A cross-encoder needs the raw
pair and a dense model needs raw text, so an arm now receives a Ctx carrying tokens AND text AND
the conversation id (for embedding caches). Lexical arms simply ignore the extra fields.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from kvmemory.components import _tok


@dataclass
class Ctx:
    """Everything an arm might need about one conversation's candidate pool."""
    conv: str
    tok: list                     # list[list[str]] tokenised docs
    txt: list                     # list[str] raw docs (what a deployed store holds)
    turn: list                    # list[int] turn index, for the recency tiebreak


# ---------------------------------------------------------------- lexical

class BM25Arm:
    name = "bm25"

    def __init__(self):
        from rank_bm25 import BM25Okapi
        self._B = BM25Okapi
        self._cache = {}

    def scores(self, query, ctx):
        if ctx.conv not in self._cache:
            self._cache[ctx.conv] = self._B(ctx.tok)
        return list(self._cache[ctx.conv].get_scores(_tok(query)))


class OverlapArm:
    """kvmemory's LexicalRouter: |set(q) & set(d)|. NEGATIVE CONTROL -- deployed nowhere."""
    name = "overlap(neg-ctrl)"

    def scores(self, query, ctx):
        q = set(_tok(query))
        return [float(len(q & set(d))) for d in ctx.tok]


def _pool_fingerprint(texts):
    """sha1 over the exact strings that were embedded.

    The doc cache is keyed on the conversation id alone, and the only thing `docs()` checked
    before trusting a cached matrix was its ROW COUNT.  That is not enough: kv_e2sets embeds
    `_utterance(s)` (the "[<ts>] <speaker>: " prefix stripped) while rA_rank with the default
    --index=text ranks `s.text` (prefix kept).  Same turns, same count, different strings.  Sharing
    a --cache-dir between the two silently scored one rendering's embeddings against the other's
    pool -- measured row-wise cosine 0.68-0.84, which is nowhere near device noise but is still
    correlated enough that nothing looks broken: self-retrieval still returns the right row and the
    format-leak gate still passes.  It moved the dense arm's delta from -0.221 to +0.071, i.e. from
    "finds values" to "prefers mentions", which would have been written up as a finding.
    Fingerprint the pool, and refuse a cache that does not match it."""
    import hashlib
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _from_pretrained_fp32(cls, path, **kw):
    """transformers renamed `torch_dtype=` to `dtype=`. Two of our hosts sat a version
    apart, and calling the wrong spelling raises -- which is the good case. The bad case would be
    dropping the argument to make the error go away: DenseArm's docstring records what reduced
    precision costs here (NaN scores that read as an ordinary weak retriever, not as an error).
    So try both spellings, then ASSERT the weights actually came back fp32."""
    import torch
    try:
        m = cls.from_pretrained(path, dtype=torch.float32, **kw)
    except TypeError as e:
        if "dtype" not in str(e):
            raise
        m = cls.from_pretrained(path, torch_dtype=torch.float32, **kw)
    assert next(m.parameters()).dtype is torch.float32, \
        f"{path} loaded as {next(m.parameters()).dtype}, not fp32"
    return m


# ---------------------------------------------------------------- dense

class DenseArm:
    """Qwen3-Embedding-0.6B cosine, the EmbeddingRouter baseline. Doc embeddings are computed once
    per conversation and cached to disk -- 5,882 turns is a few minutes on MPS, and every later arm
    (RRF, rerank) reuses them.

    PADDING SIDE IS NOT COSMETIC HERE. components/embed.py uses LEFT padding so that last-token
    pooling can read index -1. Under MPS that produces NaN in every dtype tried (fp16, bf16, fp32):
    a left-padded causal row whose only visible position is itself masked softmaxes to NaN, and the
    NaN then propagates through the whole batch. Nothing errors -- the scores come back as NaN, the
    sort degenerates to the recency tiebreak, and the arm reports a plausible-looking near-random
    ranking. It was the planted-row gate (0/200) and the format-leak probe (261.1 -> 261.1, no
    movement) that caught it, not the numbers, which looked like an ordinary weak-retriever result.
    We therefore pad RIGHT and gather at `attention_mask.sum(1)-1`, which is the same vector by
    construction and leaves no fully-masked row. Verified: MPS/fp32/right reproduces CPU/fp32
    cosines to four decimals on a fixed probe. The NaN assertion below stays in permanently."""
    name = "dense"

    def __init__(self, model_path, device=None, cache_dir=None, batch=32, max_len=512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        # RA_DEVICE moves both encoders onto a GPU. On a CPU-quota-limited host a CPU
        # cross-encoder is capped, not merely slow. Default unchanged.
        device = device or os.environ.get("RA_DEVICE") or (
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tok.padding_side = "right"         # see the class docstring -- left padding NaNs on MPS
        self.model = _from_pretrained_fp32(
            AutoModel, model_path, trust_remote_code=True,
            low_cpu_mem_usage=False).to(device).eval()
        self.device, self.batch, self.max_len = device, batch, max_len
        self.cache_dir = cache_dir
        self.QUERY_INSTRUCT = ("Given a question about a conversation, "
                               "retrieve the turns needed to answer it")
        self._doc = {}
        self._q = {}

    def _encode(self, texts, is_query):
        import numpy as np
        if is_query:
            texts = [f"Instruct: {self.QUERY_INSTRUCT}\nQuery: {t}" for t in texts]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch):
                enc = self.tok(texts[i:i + self.batch], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_len).to(self.device)
                H = self.model(**enc).last_hidden_state
                last = enc["attention_mask"].sum(1) - 1           # last REAL token, right-padded
                h = H[self.torch.arange(H.size(0), device=H.device), last]
                h = self.torch.nn.functional.normalize(h.float(), p=2, dim=-1)
                out.append(h.cpu().numpy())
        E = np.concatenate(out, 0)
        assert np.isfinite(E).all(), (
            "dense embeddings contain NaN/inf -- a NaN here does not raise, it silently degrades "
            "every ranking to the recency tiebreak. Check the padding side and the device backend.")
        return E

    def docs(self, ctx):
        """Embed the pool once per conversation.

        The planted-row gate hands us `conv#plantNN` -- the SAME pool plus one appended row. Keying
        the cache on the raw conv id there would re-embed all ~600 turns for each of the 200 planted
        items (120k forward passes for a gate that only needs one new row scored). Split the suffix
        off, reuse the base matrix, and encode ONLY the appended rows. The prefix assertion is what
        makes that safe: if the planted pool were ever anything but base + appendix, this raises
        instead of silently scoring a misaligned matrix."""
        import numpy as np
        base_conv = ctx.conv.split("#")[0]
        if ctx.conv in self._doc:
            return self._doc[ctx.conv]
        if base_conv not in self._doc:
            f = os.path.join(self.cache_dir, f"dense_docs_{base_conv}.npy") if self.cache_dir else None
            if f and os.path.exists(f):
                B = np.load(f)
                want = _pool_fingerprint(ctx.txt[:B.shape[0]])
                got = open(f + ".sha1").read().strip() if os.path.exists(f + ".sha1") else None
                assert got == want, (
                    f"{f} was built from different text than this pool "
                    f"(fingerprint {got} != {want}). A row-count match is NOT enough -- "
                    f"kv_e2sets embeds _utterance(s) and rA_rank --index=text ranks s.text. "
                    f"Delete this cache directory or give this run its own --cache-dir.")
            else:
                assert base_conv == ctx.conv, (
                    f"{ctx.conv}: the base pool for {base_conv} has not been embedded yet, so its "
                    f"size is unknown -- run the primary pass before any appended-pool gate")
                B = self._encode(ctx.txt, False)
                if f:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    np.save(f, B)
                    open(f + ".sha1", "w").write(_pool_fingerprint(ctx.txt))
            self._doc[base_conv] = B
        B = self._doc[base_conv]
        if base_conv == ctx.conv:
            assert B.shape[0] == len(ctx.txt), \
                f"stale doc cache for {base_conv}: {B.shape[0]} rows vs {len(ctx.txt)} docs"
            return B
        assert len(ctx.txt) >= B.shape[0], \
            f"{ctx.conv}: pool ({len(ctx.txt)}) is smaller than its base ({B.shape[0]})"
        D = np.concatenate([B, self._encode(ctx.txt[B.shape[0]:], False)], 0)
        self._doc[ctx.conv] = D
        return D

    def scores(self, query, ctx):
        if query not in self._q:
            self._q[query] = self._encode([query], True)[0]
        return list(self.docs(ctx) @ self._q[query])


# ---------------------------------------------------------------- fusion

class RRFArm:
    """HybridRouter's reciprocal rank fusion. Returns a SCORE (higher better) so it plugs into the
    same ranks_from() as everything else; the recency tiebreak is applied there, once."""
    def __init__(self, subs, c=60, name=None):
        self.subs = subs
        self.c = c
        self.name = name or "rrf(" + "+".join(s.name for s in subs) + ")"

    def scores(self, query, ctx):
        from kvmemory.rA_rank import ranks_from
        fused = [0.0] * len(ctx.tok)
        for s in self.subs:
            pos = ranks_from(s.scores(query, ctx), ctx.turn)
            for i, r in enumerate(pos):
                fused[i] += 1.0 / (self.c + r)
        return fused


# ---------------------------------------------------------------- cross-encoder

class RerankArm:
    """bge-reranker-v2-m3 over the base arm's top-`depth`.

    A cross-encoder is never run over a whole corpus; it reranks a retrieved head. Rows below the
    head keep their base ordering and are pushed underneath the reranked head, so the returned
    score vector still induces a TOTAL order and min-rank over M/V is always defined. `depth` is
    reported with every number -- it is the arm's main free parameter and a row the base arm buries
    below `depth` can never be rescued, which is itself part of what we are measuring."""
    def __init__(self, base, model_path, depth=50, device=None, batch=16, max_len=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.base, self.depth = base, depth
        self.name = f"rerank({base.name},d={depth})"
        # RA_DEVICE moves both encoders onto a GPU. On a CPU-quota-limited host a CPU
        # cross-encoder is capped, not merely slow. Default unchanged.
        device = device or os.environ.get("RA_DEVICE") or (
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        # fp32 everywhere. DenseArm's docstring records why: reduced precision on the MPS backend
        # returned NaN silently, and a NaN score is indistinguishable from a weak retriever in the
        # output. bge-reranker-v2-m3 is a bidirectional encoder so it has no fully-masked-row
        # problem, but the cost of fp32 here is minutes and the cost of a silent NaN is a result.
        self.model = _from_pretrained_fp32(
            AutoModelForSequenceClassification, model_path).to(device).eval()
        self.device, self.batch, self.max_len = device, batch, max_len

    def scores(self, query, ctx):
        from kvmemory.rA_rank import ranks_from
        base = self.base.scores(query, ctx)
        pos = ranks_from(base, ctx.turn)
        head = sorted(range(len(base)), key=lambda i: pos[i])[:self.depth]
        pairs = [(query, ctx.txt[i]) for i in head]
        ce = []
        with self.torch.no_grad():
            for i in range(0, len(pairs), self.batch):
                b = pairs[i:i + self.batch]
                enc = self.tok([x[0] for x in b], [x[1] for x in b], return_tensors="pt",
                               padding=True, truncation=True, max_length=self.max_len).to(self.device)
                ce += self.model(**enc).logits.float().view(-1).cpu().tolist()
        assert all(x == x and abs(x) != float("inf") for x in ce), \
            "cross-encoder returned NaN/inf -- see DenseArm's docstring on silent NaN"
        # head scores shifted above every tail score; tail keeps the base order
        lo = min(ce) if ce else 0.0
        out = [lo - 1e3 - pos[i] for i in range(len(base))]
        for i, s in zip(head, ce):
            out[i] = s
        return out


# ---------------------------------------------------------------- registry

def build_arms(names, model_dir=None, cache_dir=None, rerank_depth=50):
    """names: comma-separated. 'lex' = bm25 + overlap control."""
    reg, want = {}, [n.strip() for n in names.split(",") if n.strip()]
    need = set(want)
    if "lex" in need:
        need |= {"bm25", "overlap"}
    dense = None
    if need & {"dense", "rrf", "rerank", "rerank_dense"}:
        dense = DenseArm(os.path.join(model_dir, "Qwen3-Embedding-0.6B"), cache_dir=cache_dir)
    bm = BM25Arm()
    out = []
    for n in want:
        if n == "lex":
            out += [bm, OverlapArm()]
        elif n == "bm25":
            out.append(bm)
        elif n == "overlap":
            out.append(OverlapArm())
        elif n == "dense":
            out.append(dense)
        elif n == "rrf":
            out.append(RRFArm([bm, dense]))
        elif n == "rerank_dense":
            # the same cross-encoder over a DENSE-only head. Isolates what the fusion contributes:
            # RRF re-admits BM25's ranking, and the reranker can only reorder what its base hands it.
            out.append(RerankArm(dense, os.path.join(model_dir, "bge-reranker-v2-m3"),
                                 depth=rerank_depth))
        elif n == "rerank":
            out.append(RerankArm(RRFArm([bm, dense]),
                                 os.path.join(model_dir, "bge-reranker-v2-m3"),
                                 depth=rerank_depth))
        else:
            raise SystemExit(f"unknown arm {n!r}")
    for a in out:
        reg[a.name] = a
    return reg
