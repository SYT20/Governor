"""GeminiM2 — reasoning arm with a GENUINELY ENFORCED reasoning-token budget.

Behind the same frozen contract as MathM2 and LLMM2. The Governor cannot tell
which engine it is calling.

WHY THIS REPLACES LLMM2 AS THE PRIMARY ENGINE. Measured, not assumed:

  OpenRouter / nemotron-nano-9b-v2
    reasoning.max_tokens   IGNORED (budget 128 -> 155 spent)
    reasoning.exclude      IGNORED (budget 0   ->  81 spent)
    only total max_tokens enforced, and reasoning starves the answer:
      max_tokens=400 -> finish=length, content=None

  Gemini 2.5 Flash
    thinkingConfig.thinkingBudget  ENFORCED
      budget 0   -> thoughtsTokenCount absent, answer still produced
      budget 512 -> thoughtsTokenCount 250, answer produced
    thoughtsTokenCount reported separately from candidatesTokenCount

So here the budget constrains REASONING rather than total generation, and a
zero-budget call still answers. That removes the confound where "could not
answer" and "answered wrongly" were the same observation.

cost_units = thoughts_tokens, the resource the Governor is allocating.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from governor.gate.m2_interface import M2Result

URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "{model}:generateContent?key={key}")
MODEL = "gemini-2.5-flash"


class GeminiM2:
    name = "gemini_m2"

    def __init__(self, model: str = MODEL, api_key: str | None = None,
                 retries: int = 3):
        self.model = model
        self.key = api_key or os.environ.get("GEMINI_KEY", "")
        self.retries = retries

    def __call__(self, state: dict, reasoning_budget: float) -> M2Result:
        body = {
            "contents": [{"parts": [{"text": state["prompt"]}]}],
            "generationConfig": {
                "temperature": 0,
                "thinkingConfig": {"thinkingBudget": int(reasoning_budget)},
            },
        }
        t0 = time.time()
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    URL.format(model=self.model, key=self.key),
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.load(r)
                if "error" in d:
                    raise RuntimeError(str(d["error"])[:160])
                c = d.get("candidates", [{}])[0]
                txt = "".join(p.get("text", "")
                              for p in c.get("content", {}).get("parts", []))
                u = d.get("usageMetadata", {})
                th = float(u.get("thoughtsTokenCount") or 0)
                return M2Result(
                    result=txt.strip(),
                    reasoning_tokens=th,
                    total_tokens=float(u.get("totalTokenCount") or 0),
                    latency_s=time.time() - t0,
                    cost_units=th,
                    ok=bool(txt.strip()),
                    error="" if txt.strip() else "empty content")
            except Exception as e:                        # noqa: BLE001
                if attempt == self.retries - 1:
                    return M2Result(result=None, latency_s=time.time() - t0,
                                    cost_units=0.0, ok=False, error=str(e)[:160])
                time.sleep(2 * (attempt + 1))
        return M2Result(result=None, ok=False, error="unreachable")
