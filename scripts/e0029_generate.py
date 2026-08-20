#!/usr/bin/env python3
"""E0029 -- generate a second model's samples for the frozen problem list.

Roughly 18 hours of wall clock at the measured 8000 tokens/min, so the design
priority is not speed but survivability:

  RESUMABLE   every completed sample is appended to a JSONL cache immediately.
              Restarting skips whatever is already there. A crash, a rate-limit
              wall, or a laptop lid costs at most one sample.

  PACED       the free tier's ceiling is tokens per minute, not requests. A
              sliding-window pacer holds spend under the limit rather than
              sprinting into a 429 and backing off, which wastes the reserved
              tokens a rejected request still bills.

  HONEST      a call that ultimately fails is recorded as a failure with its
              reason, not silently dropped. A missing sample and a failed sample
              mean different things to the analysis.

The problem list and split were frozen in configs/e0029_split.json before this
script first ran. Nothing here chooses data.

Usage:
    python scripts/e0029_generate.py [--limit N] [--model M]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from governor.config import resolve_key                      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "results" / "e0029_problems.json"
CACHE = ROOT / "results" / "e0029_generations.jsonl"
URL = "https://api.groq.com/openai/v1/chat/completions"
SAMPLES = 10


class Pacer:
    """Sliding-window token budget. Waits rather than colliding with the limit."""

    def __init__(self, tokens_per_min: int = 7200):     # 10% under the 8000 ceiling
        self.limit = tokens_per_min
        self.events: collections.deque = collections.deque()

    def wait(self, cost: int) -> None:
        while True:
            now = time.time()
            while self.events and now - self.events[0][0] > 60:
                self.events.popleft()
            spent = sum(c for _, c in self.events)
            if spent + cost <= self.limit:
                return
            time.sleep(max(0.5, 60 - (now - self.events[0][0])))

    def charge(self, cost: int) -> None:
        self.events.append((time.time(), cost))


def build_prompt(p: dict) -> str:
    starter = p.get("starter_code", "")
    if starter:
        return (f"{p['question_content']}\n\nComplete this solution:\n"
                f"```python\n{starter}\n```\n"
                "Respond with ONLY the complete ```python code block.")
    return (f"{p['question_content']}\n\nRead input from stdin and write the answer "
            "to stdout. Respond with ONLY a ```python code block.")


def extract_code(text: str) -> str:
    """Take the last fenced python block; models often narrate before the answer."""
    if "```" not in text:
        return text.strip()
    blocks, parts = [], text.split("```")
    for i in range(1, len(parts), 2):
        b = parts[i]
        if b.startswith("python"):
            b = b[len("python"):]
        blocks.append(b.strip())
    return blocks[-1] if blocks else text.strip()


def call(key: str, model: str, prompt: str, pacer: Pacer, tries: int = 5):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_completion_tokens": 2500, "temperature": 0.2}).encode()
    est = min(2500, len(prompt)//3 + 1200)
    for attempt in range(tries):
        pacer.wait(est)
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            u = d.get("usage", {})
            pacer.charge(int(u.get("total_tokens", est)))
            msg = d["choices"][0]["message"]
            return {"ok": True, "content": msg.get("content") or "",
                    "tokens": int(u.get("completion_tokens", 0)),
                    "total_tokens": int(u.get("total_tokens", 0))}
        except urllib.error.HTTPError as e:
            pacer.charge(est)                    # a rejected request still bills
            if e.code == 429:
                time.sleep(min(60, 5 * 2 ** attempt)); continue
            if e.code in (500, 502, 503) and attempt < tries - 1:
                time.sleep(3 * 2 ** attempt); continue
            return {"ok": False, "error": f"HTTP {e.code}"}
        except Exception as e:                   # noqa: BLE001
            if attempt < tries - 1:
                time.sleep(3 * 2 ** attempt); continue
            return {"ok": False, "error": type(e).__name__}
    return {"ok": False, "error": "retries exhausted"}


def done_keys() -> set:
    if not CACHE.exists():
        return set()
    out = set()
    with open(CACHE) as f:
        for line in f:
            try:
                d = json.loads(line)
                out.add((d["qid"], d["sample"]))
            except Exception:                     # noqa: BLE001
                continue                          # a torn final line is fine
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    args = ap.parse_args()

    key = resolve_key("groq")
    problems = json.loads(PROBLEMS.read_text())
    if args.limit:
        problems = problems[:args.limit]
    have = done_keys()
    todo = [(p, s) for p in problems for s in range(SAMPLES)
            if (p["qid"], s) not in have]

    print(f"problems={len(problems)} samples={SAMPLES} "
          f"already_done={len(have)} todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return 0

    pacer, t0, spent, fails = Pacer(), time.perf_counter(), 0, 0
    with open(CACHE, "a") as out:
        for i, (p, s) in enumerate(todo, 1):
            r = call(key, args.model, build_prompt(p), pacer)
            rec = {"qid": p["qid"], "sample": s, "model": args.model}
            if r["ok"]:
                rec.update(code=extract_code(r["content"]), tokens=r["tokens"],
                           total_tokens=r["total_tokens"], ok=True)
                spent += r["total_tokens"]
            else:
                rec.update(ok=False, error=r["error"], code="", tokens=0,
                           total_tokens=0)
                fails += 1
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if i % 50 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(todo)}  {el/60:>6.1f}min  "
                      f"eta {el/i*(len(todo)-i)/3600:>5.2f}h  "
                      f"tokens {spent/1e6:.2f}M  failures {fails}", flush=True)

    print(f"done: {len(todo)} attempted, {fails} failed, {spent/1e6:.2f}M tokens, "
          f"{(time.perf_counter()-t0)/3600:.2f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
