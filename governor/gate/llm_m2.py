"""LLMM2 — the LLM reasoning arm behind the FROZEN M2 contract.

The Governor must not learn that this is an LLM. Same signature as MathM2:
    M2(state, reasoning_budget) -> M2Result

BUDGET MECHANISM -- MEASURED, NOT ASSUMED. My first version passed
`reasoning.max_tokens` and `reasoning.exclude`. Both are IGNORED by this
provider:

    budget 0   -> 81 reasoning tokens spent anyway (exclude ignored)
    budget 128 -> 155 reasoning tokens spent (cap exceeded)

What IS honoured is total `max_tokens`:

    max_tokens=400   finish=length  content=None    reasoning=242  starved
    max_tokens=1200  finish=stop    content='3196'  reasoning=529  correct

So the budget lever is total max_tokens. Reasoning consumes it first and the
answer is emitted last, which means an under-budgeted call returns EMPTY -- the
model thinks and then runs out of room to answer. A starved call is a FAILED
call (ok=False), never a zero-utility success, or the curve would confuse
"could not answer" with "answered wrongly".

`reasoning_tokens` from usage remains the observable COST, which is what the
contract requires.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from governor.gate.m2_interface import M2Result

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-nano-9b-v2:free"
# ANSWER_HEADROOM removed: the provider ignores reasoning.max_tokens, so there
# is no separate reasoning cap to add headroom on top of.


class LLMM2:
    name = "llm_m2"

    def __init__(self, model: str = MODEL, api_key: str | None = None,
                 retries: int = 3):
        self.model = model
        self.key = api_key or os.environ.get("OR_KEY", "")
        self.retries = retries

    def __call__(self, state: dict, reasoning_budget: float) -> M2Result:
        prompt = state["prompt"]
        rb = int(reasoning_budget)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content":
                 "Answer with only the final number. No words, no units."},
                {"role": "user", "content": prompt}],
            "max_tokens": rb,          # the ONLY lever this provider honours
            "temperature": 0.0,
        }

        t0 = time.time()
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    URL, data=json.dumps(body).encode(),
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.load(r)
                if "error" in d:
                    raise RuntimeError(str(d["error"])[:160])
                msg = d["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                u = d.get("usage", {})
                rt = float(u.get("completion_tokens_details", {})
                           .get("reasoning_tokens", 0))
                return M2Result(
                    result=content,
                    reasoning_tokens=rt,
                    total_tokens=float(u.get("total_tokens", 0)),
                    latency_s=time.time() - t0,
                    cost_units=rt,
                    ok=bool(content),
                    error="" if content else "empty content (reasoning starved answer)")
            except Exception as e:                       # noqa: BLE001
                if attempt == self.retries - 1:
                    return M2Result(result=None, latency_s=time.time() - t0,
                                    cost_units=0.0, ok=False, error=str(e)[:160])
                time.sleep(2 * (attempt + 1))
        return M2Result(result=None, ok=False, error="unreachable")
