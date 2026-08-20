#!/usr/bin/env python3
"""Measure sequential vs batched generation on THIS machine, then pick.

Batching is usually assumed to win on a GPU. Measured on Apple Silicon via MLX it
LOST -- 0.96x, slightly slower than sequential:

    sequential   152.0s   2500 tokens   16.4 tok/s
    batched      157.8s   3100 tokens   19.6 tok/s

The token counts show why. A batch runs until its LONGEST sequence finishes, so
short completions spend steps generating padding. Throughput rose 19% and wall
clock did not move, because the batch did 24% more work. With this workload's
length distribution -- median 212, p95 1128 -- that waste is severe.

MLX on unified memory is bandwidth-bound at batch size 1, so there is no idle
compute for batching to fill; a CUDA GPU is the opposite. Rather than extrapolate
across that difference, this script measures both paths on whatever hardware it
finds and reports which to use.

It also checks something a timing benchmark alone would miss: whether batched
sampling still produces INDEPENDENT samples. In the MLX run, 8/8 outputs were
distinct sequentially against 6/8 batched. If a shared sampler correlates draws,
the ten samples per problem stop being independent and the allocation measurement
is corrupted -- a correctness failure that looks like a speedup.

    python scripts/colab_batch_bench.py [--n 8] [--cap 1024]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]

MIN_SPEEDUP = 1.15          # below this, batching is not worth its complexity
MIN_DISTINCT_FRACTION = 0.9  # batched samples must stay essentially as diverse


def build_prompts(n: int) -> list[dict]:
    probs = json.loads((ROOT / "results" / "e0029_problems.json").read_text())
    out, i = [], 0
    while len(out) < n:
        out.append(probs[i % len(probs)])
        i += 1
    return out


def prompt_text(p: dict) -> str:
    return (f"{p['question_content'][:4000]}\n\nWrite a Python program that reads from "
            "stdin and writes the answer to stdout. Respond with ONLY a ```python "
            "code block.")


def run_torch(n: int, cap: int, model_id: str) -> dict:
    """The CUDA path: transformers, left-padded batch against a sequential loop."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu")
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    # Decoder-only models must be LEFT padded; right padding makes the model
    # continue from pad tokens and emit garbage that still looks like output.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.perf_counter()
    import inspect
    _k = ({"dtype": dtype}
          if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
          else {"torch_dtype": dtype})
    model = AutoModelForCausalLM.from_pretrained(model_id, **_k).to(dev)
    load_s = time.perf_counter() - t0
    model.eval()

    probs = build_prompts(n)
    texts = [tok.apply_chat_template([{"role": "user", "content": prompt_text(p)}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in probs]

    gen_kw = dict(max_new_tokens=cap, do_sample=True, temperature=0.7, top_p=0.95,
                  pad_token_id=tok.pad_token_id)

    torch.manual_seed(0)
    t0 = time.perf_counter()
    seq_out = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model.generate(**enc, **gen_kw)
        seq_out.append(tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    seq_s = time.perf_counter() - t0

    torch.manual_seed(0)
    t0 = time.perf_counter()
    enc = tok(texts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        o = model.generate(**enc, **gen_kw)
    bat_out = [tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
               for row in o]
    bat_s = time.perf_counter() - t0

    peak = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else None
    return {"backend": "torch", "device": dev, "load_s": load_s,
            "sequential_s": seq_s, "batched_s": bat_s,
            "sequential_out": seq_out, "batched_out": bat_out,
            "tok": tok, "peak_gpu_gb": peak}


def run_mlx(n: int, cap: int, model_id: str) -> dict:
    import mlx.core as mx
    from mlx_lm import batch_generate, generate, load
    from mlx_lm.sample_utils import make_sampler

    t0 = time.perf_counter()
    model, tok = load(model_id)
    load_s = time.perf_counter() - t0
    probs = build_prompts(n)
    texts = [tok.apply_chat_template([{"role": "user", "content": prompt_text(p)}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in probs]

    mx.random.seed(0)
    t0 = time.perf_counter()
    seq_out = [generate(model, tok, prompt=t, max_tokens=cap,
                        sampler=make_sampler(temp=0.7), verbose=False) for t in texts]
    seq_s = time.perf_counter() - t0

    mx.random.seed(0)
    t0 = time.perf_counter()
    res = batch_generate(model, tok, prompts=[tok.encode(t) for t in texts],
                         max_tokens=cap, sampler=make_sampler(temp=0.7), verbose=False)
    bat_s = time.perf_counter() - t0
    bat_out = res.texts if hasattr(res, "texts") else list(res)

    return {"backend": "mlx", "device": "apple-silicon", "load_s": load_s,
            "sequential_s": seq_s, "batched_s": bat_s,
            "sequential_out": seq_out, "batched_out": bat_out,
            "tok": tok, "peak_gpu_gb": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--model", default="")
    ap.add_argument("--json", default="results/batch_bench.json")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "configs" / "colab_model.json").read_text())
    try:
        import torch                                        # noqa: F401
        model_id = args.model or cfg["model_name"]
        r = run_torch(args.n, args.cap, model_id)
    except ImportError:
        model_id = args.model or "mlx-community/Qwen3-1.7B-4bit"
        r = run_mlx(args.n, args.cap, model_id)

    tok = r.pop("tok")
    ntok = lambda xs: sum(len(tok.encode(x)) for x in xs)
    s_tok, b_tok = ntok(r["sequential_out"]), ntok(r["batched_out"])
    speedup = r["sequential_s"] / r["batched_s"] if r["batched_s"] else 0.0
    d_seq = len(set(r["sequential_out"])) / len(r["sequential_out"])
    d_bat = len(set(r["batched_out"])) / len(r["batched_out"])

    print(f"\n  backend {r['backend']} on {r['device']}   model {model_id}")
    print(f"  loaded in {r['load_s']:.1f}s"
          + (f"   peak GPU {r['peak_gpu_gb']:.2f} GB" if r["peak_gpu_gb"] else ""))
    print(f"\n  {'mode':<12}{'wall_s':>9}{'tokens':>9}{'tok/s':>9}{'s/sample':>10}")
    print("  " + "-" * 50)
    print(f"  {'sequential':<12}{r['sequential_s']:>9.1f}{s_tok:>9}"
          f"{s_tok/r['sequential_s']:>9.1f}{r['sequential_s']/args.n:>10.2f}")
    print(f"  {'batched':<12}{r['batched_s']:>9.1f}{b_tok:>9}"
          f"{b_tok/r['batched_s']:>9.1f}{r['batched_s']/args.n:>10.2f}")
    print(f"\n  speedup {speedup:.2f}x on a batch of {args.n}")
    print(f"  padding waste: batched generated {b_tok/max(s_tok,1):.2f}x the tokens")
    print(f"  distinct outputs: sequential {d_seq:.0%}  batched {d_bat:.0%}")

    verdict, why = "SEQUENTIAL", []
    if speedup < MIN_SPEEDUP:
        why.append(f"speedup {speedup:.2f}x below the {MIN_SPEEDUP}x threshold")
    if d_bat < d_seq * MIN_DISTINCT_FRACTION:
        why.append(f"batched sample diversity {d_bat:.0%} vs sequential {d_seq:.0%} "
                   f"-- samples may not be independent")
    if not why:
        verdict = "BATCHED"

    total = 4750
    est_h = (r["batched_s"] if verdict == "BATCHED" else r["sequential_s"]) / args.n * total / 3600
    print(f"\n  VERDICT: use {verdict}")
    for w in why:
        print(f"    - {w}")
    print(f"  projected {total} samples: {est_h:.1f} h")

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "backend": r["backend"], "device": r["device"], "model": model_id,
        "n": args.n, "cap": args.cap, "load_s": r["load_s"],
        "sequential_s": r["sequential_s"], "batched_s": r["batched_s"],
        "sequential_tokens": s_tok, "batched_tokens": b_tok,
        "speedup": speedup, "padding_waste": b_tok / max(s_tok, 1),
        "distinct_sequential": d_seq, "distinct_batched": d_bat,
        "peak_gpu_gb": r["peak_gpu_gb"], "verdict": verdict, "reasons": why,
        "projected_hours_4750": est_h}, indent=1))
    print(f"  wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
