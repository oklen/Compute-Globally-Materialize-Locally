"""One chat call to an OpenAI-compatible LLM endpoint.

Used where the experiments need a model other than the local reader:
  * LLM-written hard negatives  (kv_e3negs, kv_e2negs, kv_hnegs)
  * loose answer-span labels for the real-dialog retrieval test  (rA_loose)
  * the cross-family re-judge of the E2 answers  (kv_e2_rejudge)
  * the graph-index ingestion arm reads the same variables  (rA_graph, through graphiti)

Configure it with environment variables; there are no defaults, so a missing one fails fast:

    JUDGE_BASE_URL    base URL of any OpenAI-compatible server, e.g. https://api.openai.com/v1
    JUDGE_API_KEY     the key for that server
    JUDGE_MODEL       the model name on that server
    JUDGE_EXTRA_BODY  optional JSON merged into the request body, e.g.
                      '{"thinking": {"type": "disabled"}}' for servers with a reasoning switch

The reported runs used a commercial chat model from a different family than the Qwen3 reader,
with reasoning disabled: with reasoning on, that endpoint spent 80-90 s per request and returned the
reasoning trace instead of the answer. Any OpenAI-compatible endpoint can stand in, but the numbers
that depend on it -- the LLM-written negatives and the cross-family re-judge -- will differ.
"""
from __future__ import annotations

import json
import os
import urllib.request


def _config():
    missing = [v for v in ("JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL") if not os.environ.get(v)]
    if missing:
        raise RuntimeError("Set %s to an OpenAI-compatible endpoint (see kvmemory/llm_api.py)."
                           % ", ".join(missing))
    extra = json.loads(os.environ.get("JUDGE_EXTRA_BODY") or "{}")
    return os.environ["JUDGE_BASE_URL"].rstrip("/"), os.environ["JUDGE_API_KEY"], os.environ["JUDGE_MODEL"], extra


def chat(prompt: str, timeout: float = 120) -> str:
    """Send one user message and return the assistant text ("" if the server returned none)."""
    base, key, model, extra = _config()
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    body.update(extra)
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.load(r)
    choices = j.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""
