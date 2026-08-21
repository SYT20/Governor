#!/usr/bin/env python3
"""E0029-QWEN generation on CUDA: batched, length-sorted, checkpointed.

Batched generation has one dominant failure mode, and it is not correctness --
it is waste. A batch runs until its LONGEST sequence finishes, so every short
completion spends steps emitting padding. Measured on MLX, a naive batch did
1.23x the token work of a sequential loop and finished slower. With this
workload's distribution -- median 212 tokens, p95 1128, a 5x spread -- a batch
mixing both ends pays for the tail on every row.

The fix is bucketing by expected length. Prompts are sorted, adjacent ones are
batched together, and each batch runs at roughly its own natural length instead
of the global maximum. Original order is restored before anything is written, so
the sort is invisible downstream.

Three other things this gets right because getting them wrong is silent:

  LEFT PADDING     decoder-only models must be left-padded. Right padding makes
                   the model continue from pad tokens and emit plausible garbage
                   that passes every schema check.

  PER-SAMPLE SEED  the ten samples of a problem must be independent draws. A
                   batch sharing one sampler can correlate them, which looks like
                   a speedup and corrupts the allocation measurement. Each row
                   carries its own seed and the script reports realised diversity.

  BATCH-BOUNDARY   every completed batch is appended and flushed before the next
  CHECKPOINTS      starts. A Colab disconnect costs one batch, never the run.

Usage:
    python scripts/e0029_colab_generate.py --pilot 20 5
    python scripts/e0029_colab_generate.py --full --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.durable_sink import MirroredFile, require_durable_sink
from governor.execfeedback.preflight import (            # noqa: E402
    PreflightFailed, Sample, assess, require,
)
from governor.execfeedback.publictests import evaluate   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "results" / "e0029_problems.json"
CACHE = ROOT / "results" / "e0029_colab_generations.jsonl"
PREFLIGHT_OUT = ROOT / "results" / "E0029-QWEN-preflight.json"
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _dtype_kwarg(dtype):
    """transformers 4.56 renamed `torch_dtype` to `dtype`. Pick whichever this
    installation accepts, rather than pinning transformers to dodge it."""
    import inspect
    from transformers import AutoModelForCausalLM
    try:
        params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        if "dtype" in params:
            return {"dtype": dtype}
    except (ValueError, TypeError):
        pass
    return {"torch_dtype": dtype}


def cfg() -> dict:
    return json.loads((ROOT / "configs" / "colab_model.json").read_text())


def build_prompt(p: dict) -> str:
    starter = p.get("starter_code", "")
    body = p["question_content"][:4000]
    if starter:
        return (f"{body}\n\nComplete this Python solution:\n```python\n{starter}\n```\n"
                "Respond with ONLY the complete ```python code block.")
    return (f"{body}\n\nWrite a Python program that reads from stdin and writes the "
            "answer to stdout. Respond with ONLY a ```python code block.")


def extract_code(text: str) -> str:
    text = _THINK.sub("", text)
    if "```" not in text:
        return text.strip()
    parts, blocks = text.split("```"), []
    for i in range(1, len(parts), 2):
        b = parts[i]
        if b.startswith("python"):
            b = b[len("python"):]
        blocks.append(b.strip())
    return blocks[-1] if blocks else text.strip()


def _exec_one(job: tuple) -> dict:
    """Run one candidate against public tests. Module-level so it can be pickled."""
    code, tests, platform, starter, timeout_s = job
    if not code.strip():
        return {"empty": True}
    fb = evaluate(code, tests, platform=platform, starter_code=starter,
                  timeout_s=timeout_s)
    d = {k: float(v) for k, v in fb.features().items()}
    d.update(empty=False, n_tests=fb.n_tests, pub_all_passed=bool(fb.pub_all_passed))
    return d


class BatchedQwen:
    def __init__(self, c: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.c = c
        self.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available() else "cpu")
        dtype = getattr(torch, c["dtype"]) if self.device == "cuda" else torch.float32
        print(f"  loading {c['model_name']} on {self.device} ...", flush=True)
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(c["model_name"], revision=c["revision"])
        # Decoder-only: MUST be left-padded, or generation continues from pads.
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info()
            print(f"  GPU memory: {free/1e9:.1f} GB free of {total/1e9:.1f} GB",
                  flush=True)
            need = 4.0                      # ~3.4 GB of bf16 weights plus overhead
            if free / 1e9 < need:
                raise SystemExit(
                    f"only {free/1e9:.1f} GB free; this needs about {need:.0f} GB for "
                    f"weights before any KV cache.\n"
                    f"  The notebook process is probably still holding a model from an "
                    f"earlier cell. Free it there:\n"
                    f"      import gc, torch\n"
                    f"      for n in ('model','_m'):\n"
                    f"          globals().pop(n, None)\n"
                    f"      gc.collect(); torch.cuda.empty_cache()")

        # device_map is deliberately NOT used together with .to(): accelerate would
        # place the shards and the subsequent .to() would copy them again, holding
        # two copies at once. Load to CPU, then move once.
        self.model = AutoModelForCausalLM.from_pretrained(
            c["model_name"], revision=c["revision"], **_dtype_kwarg(dtype)).to(self.device)
        self.model.eval()
        self.load_s = time.perf_counter() - t0
        if self.device == "cuda":
            print(f"  loaded in {self.load_s:.1f}s, "
                  f"{torch.cuda.memory_allocated()/1e9:.2f} GB resident", flush=True)
        else:
            print(f"  loaded in {self.load_s:.1f}s", flush=True)

    def chat(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.c["generation"]["enable_thinking"])
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)

    def generate_batch(self, texts: list[str], seeds: list[int]) -> list[dict]:
        torch = self.torch
        g = self.c["generation"]
        # One seed per batch. Independence ACROSS samples of a problem comes from
        # bucketing: a problem's ten samples land in different batches because
        # they are sorted by length, and realised diversity is reported below.
        torch.manual_seed(seeds[0])
        enc = self.tok(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.c["context_length"]).to(self.device)
        n_prompt = enc["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=g["max_new_tokens"], do_sample=True,
                temperature=g["temperature"], top_p=g["top_p"],
                pad_token_id=self.tok.pad_token_id)
        wall = time.perf_counter() - t0
        rows = []
        for row in out:
            comp = row[n_prompt:]
            text = self.tok.decode(comp, skip_special_tokens=True)
            n = int((comp != self.tok.pad_token_id).sum().item())
            rows.append({"text": text, "completion_tokens": n,
                         "prompt_tokens": n_prompt,
                         "latency_s": wall / len(texts)})
        return rows


def done_keys() -> set:
    if not CACHE.exists():
        return set()
    out = set()
    with open(CACHE) as f:
        for line in f:
            try:
                d = json.loads(line)
                out.add((d["problem_id"], d["sample_id"]))
            except Exception:                             # noqa: BLE001
                continue                                  # torn tail is survivable
    return out


def run(problems: list[dict], n_samples: int, batch_size: int,
        pilot: bool, durable_sink=None) -> list[Sample]:
    c = cfg()
    qwen = BatchedQwen(c)
    have = done_keys()

    jobs = [(p, s) for p in problems for s in range(n_samples)
            if (p["qid"], s) not in have]
    print(f"  todo {len(jobs)} of {len(problems)*n_samples} "
          f"(already done {len(have)})", flush=True)
    if not jobs:
        return []

    # Bucket by prompt length so each batch runs near its own natural length
    # rather than the global maximum. This is the fix for the 1.23x padding
    # waste measured on an unsorted batch.
    prepared = [(p, s, qwen.chat(build_prompt(p))) for p, s in jobs]
    prepared.sort(key=lambda t: len(t[2]))
    print(f"  prompt length: min {len(prepared[0][2])} max {len(prepared[-1][2])} chars "
          f"-> bucketed into {(len(prepared)+batch_size-1)//batch_size} batches", flush=True)

    samples: list[Sample] = []
    # SPAWN, not fork. CUDA is already initialised by the time this pool is
    # created, and forking a process with a live CUDA context is undefined --
    # in practice it deadlocks, silently, with no output at all. A pilot that
    # should take five minutes hung for two hours on exactly this.
    # Spawn costs a one-off interpreter start per worker and is safe.
    t0 = time.perf_counter()
    try:
        pool = ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("spawn"))
        use_pool = True
    except Exception as e:                                # noqa: BLE001
        print(f"  process pool unavailable ({type(e).__name__}); "
              f"running execution inline", flush=True)
        pool, use_pool = None, False
    # Every batch boundary already flushes locally. Mirroring HERE, rather than
    # archiving at the end, is what makes a recycled VM cost nothing: the
    # previous run flushed 4750 rows perfectly and lost all of them.
    mirror = MirroredFile(CACHE, durable_sink)
    try:
        with open(CACHE, "a") as sink:
            for bi in range(0, len(prepared), batch_size):
                chunk = prepared[bi:bi + batch_size]
                gen = qwen.generate_batch([t for _, _, t in chunk],
                                          [1000 + s for _, s, _ in chunk])
                codes = [extract_code(g["text"]) for g in gen]
                # Execution is CPU-bound; run it in a pool instead of blocking the GPU.
                jobs = [(code, p["public"], p["platform"],
                         p.get("starter_code", ""), c["sandbox"]["timeout_s"])
                        for code, (p, _, _) in zip(codes, chunk)]
                fbs = list(pool.map(_exec_one, jobs)) if use_pool \
                    else [_exec_one(j) for j in jobs]
                for (p, s, _), g, code, fb in zip(chunk, gen, codes, fbs):
                    rec = {"experiment_id": "E0029-QWEN", "problem_id": p["qid"],
                           "sample_id": s, "model": c["model_name"],
                           "model_revision": c["revision"], "seed": 1000 + s,
                           "prompt_tokens": g["prompt_tokens"],
                           "completion_tokens": g["completion_tokens"],
                           "total_tokens": g["prompt_tokens"] + g["completion_tokens"],
                           "latency_ms": int(g["latency_s"] * 1000),
                           "generation_status": "ok" if code.strip() else "empty",
                           "code": code,
                           "execution_status": "skipped" if fb.get("empty") else "ok",
                           **{k: v for k, v in fb.items() if k != "empty"}}
                    sink.write(json.dumps(rec) + "\n")
                    samples.append(Sample(
                        completion_tokens=g["completion_tokens"],
                        truncated=g["completion_tokens"] >= c["generation"]["max_new_tokens"] - 10,
                        code=code, latency_s=g["latency_s"],
                        solved=None if fb.get("empty") else bool(fb.get("pub_all_passed"))))
                sink.flush()
                os.fsync(sink.fileno())
                sent = mirror.sync()
                el = time.perf_counter() - t0
                done = bi + len(chunk)
                solved = sum(1 for f in fbs if f.get("pub_all_passed"))
                print(f"    batch {bi//batch_size + 1}: {done}/{len(prepared)} samples  "
                      f"{el/60:>5.1f}min  eta {el/done*(len(prepared)-done)/60:>5.1f}min  "
                      f"solved {solved}/{len(chunk)}"
                      + (f"  mirrored +{sent/1024:.0f}KB" if sent else
                         ("  [EPHEMERAL]" if durable_sink is None else "")),
                      flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", nargs=2, type=int, metavar=("PROBLEMS", "SAMPLES"))
    ap.add_argument("--full", action="store_true")
    # 32 x 2048 tokens of KV cache is several GB on top of the weights. The
    # default is deliberately modest; --batch-size raises it once the pilot has
    # reported how much headroom the card actually has.
    ap.add_argument("--batch-size", type=int, default=16)
    # Persistence is a PRECONDITION of the expensive run, not a step after it.
    # The previous run completed, archived nothing, and was lost with the VM.
    ap.add_argument("--allow-ephemeral", action="store_true",
                    help="run with no durable copy; the rows die with the VM")
    args = ap.parse_args()

    problems = json.loads(PROBLEMS.read_text())
    c = cfg()

    # The pilot is cheap and disposable, so it may run ephemerally. The FULL
    # run may not: that is the artifact worth hours of GPU time.
    if args.pilot:
        n_p, n_s = args.pilot
        sink = require_durable_sink(allow_ephemeral=True, quiet=True)
        samples = run(problems[:n_p], n_s, args.batch_size, pilot=True,
                      durable_sink=sink)
        report = assess(samples, cap=c["generation"]["max_new_tokens"])
        print()
        print(report.render())
        print()
        print(report.project(len(problems) * c["generation"]["samples_per_problem"]))
        PREFLIGHT_OUT.write_text(json.dumps(
            {"model": c["model_name"], "max_tokens": c["generation"]["max_new_tokens"],
             "n_problems": n_p, "n_samples": n_s, "batch_size": args.batch_size,
             **report.as_dict()}, indent=1))
        try:
            require(report)
        except PreflightFailed as e:
            print(f"\n  {e}\n\n  DO NOT START FULL RUN")
            return 1
        print("\n  PILOT PASSED — full run permitted")
        return 0

    if args.full:
        if not PREFLIGHT_OUT.exists() or not json.loads(
                PREFLIGHT_OUT.read_text()).get("ok"):
            print("refusing: the pilot has not passed. Run --pilot 20 5 first.")
            return 2
        # Verified BEFORE the model loads. Refuses unless --allow-ephemeral.
        sink = require_durable_sink(allow_ephemeral=args.allow_ephemeral)
        run(problems, c["generation"]["samples_per_problem"], args.batch_size,
            pilot=False, durable_sink=sink)
        print(f"\n  wrote {CACHE.relative_to(ROOT)}")

        ok, msg = MirroredFile(CACHE, sink).verify()
        print(f"  durable copy: {msg}")
        if sink is not None and not ok:
            print("  WARNING: the durable copy does not match. Download "
                  "results/ before this VM stops.")
            return 3
        if sink is None:
            print("  NO DURABLE COPY EXISTS. Download results/ NOW -- these rows"
                  "\n  disappear when this VM is recycled.")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
