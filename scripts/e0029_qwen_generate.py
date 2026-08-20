#!/usr/bin/env python3
"""E0029-QWEN -- cross-backend generation with the local MLX Qwen backend.

Not a replication of the gpt-oss preregistration. That run was stopped because a
token cap was manufacturing failures, and switching backend changes the
capability regime, so calling this a replication would misstate the estimand.
It asks a different and arguably better question:

    is the allocation mechanism tied to one model's quirks, or does it survive a
    backend change?

Everything except the reasoning backend is held fixed: benchmark family,
allocation formulation, feature boundary, target, grouped CV, calibration and
evaluation discipline, resource accounting.

Local generation removes the actual bottleneck. The gpt-oss route was limited by
a provider's 8000 tokens/min, not by anything scientific; MLX has no rate limit,
so the cap can be set by what the model needs rather than by what a free tier
allows.

    --preflight N M   generate M samples on N problems, assess, and STOP.
                      A full run refuses to start until this passes.

Usage:
    python scripts/e0029_qwen_generate.py --preflight 20 5
    python scripts/e0029_qwen_generate.py --full
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from governor.execfeedback.preflight import (            # noqa: E402
    PreflightFailed, Sample, assess, require,
)
from governor.execfeedback.publictests import evaluate   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "results" / "e0029_problems.json"
CACHE = ROOT / "results" / "e0029_qwen_generations.jsonl"
PREFLIGHT_OUT = ROOT / "results" / "E0029-QWEN-preflight.json"
MODEL = "mlx-community/Qwen3-1.7B-4bit"
MAX_TOKENS = 3072                 # set by what the model needs, not a quota
SAMPLES = 10

_THINK = re.compile(r"<think>.*?</think>", re.S)


def build_prompt(p: dict) -> str:
    starter = p.get("starter_code", "")
    body = p["question_content"][:4000]
    if starter:
        return (f"{body}\n\nComplete this Python solution:\n```python\n{starter}\n```\n"
                "Respond with ONLY the complete ```python code block.")
    return (f"{body}\n\nWrite a Python program that reads from stdin and writes the "
            "answer to stdout. Respond with ONLY a ```python code block.")


def extract_code(text: str) -> str:
    text = _THINK.sub("", text)          # Qwen3 emits reasoning in <think> tags
    if "```" not in text:
        return text.strip()
    parts, blocks = text.split("```"), []
    for i in range(1, len(parts), 2):
        b = parts[i]
        if b.startswith("python"):
            b = b[len("python"):]
        blocks.append(b.strip())
    return blocks[-1] if blocks else text.strip()


class Qwen:
    def __init__(self, model_id: str = MODEL):
        from mlx_lm import load
        print(f"  loading {model_id} ...", flush=True)
        t0 = time.perf_counter()
        self.model, self.tok = load(model_id)
        print(f"  loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    def generate(self, prompt: str, temperature: float, seed: int):
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        mx.random.seed(seed)
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        t0 = time.perf_counter()
        out = generate(self.model, self.tok, prompt=text, max_tokens=MAX_TOKENS,
                       sampler=make_sampler(temp=temperature), verbose=False)
        latency = time.perf_counter() - t0
        n_tok = len(self.tok.encode(out))
        return out, n_tok, latency


def run_preflight(n_problems: int, n_samples: int) -> int:
    problems = json.loads(PROBLEMS.read_text())[:n_problems]
    qwen = Qwen()
    samples, t0 = [], time.perf_counter()
    for i, p in enumerate(problems, 1):
        for s in range(n_samples):
            out, n_tok, lat = qwen.generate(build_prompt(p), 0.7, seed=1000 + s)
            code = extract_code(out)
            solved = None
            if code.strip():
                fb = evaluate(code, p["public"], platform=p["platform"],
                              starter_code=p.get("starter_code", ""), timeout_s=6.0)
                solved = bool(fb.pub_all_passed)
            samples.append(Sample(completion_tokens=n_tok,
                                  truncated=n_tok >= MAX_TOKENS - 10,
                                  code=code, latency_s=lat, solved=solved))
        el = time.perf_counter() - t0
        print(f"    {i}/{len(problems)} problems  {el/60:>5.1f}min  "
              f"eta {el/i*(len(problems)-i)/60:>5.1f}min", flush=True)

    report = assess(samples, cap=MAX_TOKENS)
    print()
    print(report.render())
    print()
    print(report.project(len(json.loads(PROBLEMS.read_text())) * SAMPLES))
    PREFLIGHT_OUT.write_text(json.dumps(
        {"model": MODEL, "max_tokens": MAX_TOKENS, "n_problems": n_problems,
         "n_samples": n_samples, **report.as_dict()}, indent=1))
    print(f"\n  wrote {PREFLIGHT_OUT.relative_to(ROOT)}")
    try:
        require(report)
    except PreflightFailed as e:
        print(f"\n  {e}")
        return 1
    print("\n  PREFLIGHT PASSED -- a full run is permitted")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", nargs=2, type=int, metavar=("PROBLEMS", "SAMPLES"))
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.preflight:
        return run_preflight(*args.preflight)

    if args.full:
        if not PREFLIGHT_OUT.exists():
            print("refusing: no preflight on record. Run --preflight first.")
            return 2
        rep = json.loads(PREFLIGHT_OUT.read_text())
        if not rep.get("ok"):
            print(f"refusing: preflight failed -- {rep.get('problems')}")
            return 2
        print("full run not started by this commit; preflight gate is in place.")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
