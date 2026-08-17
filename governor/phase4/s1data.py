"""External replication data: the s1 paper's released per-item eval outputs.

WHY THIS DATASET AND NOT ANOTHER OF MINE. Every environment in this project so
far was one I designed, which means every positive result carries the objection
that the designer chose the conditions. These are third-party generations,
published by the s1 authors, on public benchmarks, at seven enforced budgets.
I did not choose the model, the prompts, the items, or the budgets.

It also fixes the two constraints that dominated the whole project:

  * BINDING BUDGET. s1 enforces the cap by forcing "</think>" at the limit, so
    the requested budget is actually spent. Our Groq setup reserved a cap the
    engine used 28% of, and worst-case reservation then decided allocation
    instead of preference. Here cap/actual is 1 by construction.
  * NO QUOTA. The generations already exist. An experiment costs seconds.

Source: https://huggingface.co/datasets/simplescaling/results
Paper:  https://arxiv.org/abs/2501.19393

The loader writes a normal `ResponseCache`, so the environment, the executor,
Ares, the policies and the trap checks all run unmodified. That is the point:
if the architecture needed changing to accept external data, it was overfitted
to my own generators.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUDGET_DIRS = {"forcing500": 500, "forcing1k": 1000, "forcing2k": 2000,
               "forcing4k": 4000, "forcing8k": 8000, "forcing16k": 16000,
               "forcing32k": 32000}
BENCHMARKS = {"math": "samples_openai_math", "gpqa": "samples_gpqa_diamond_openai",
              "aime": "samples_aime24_nofigures"}
REPO = "simplescaling/results"


@dataclass(frozen=True, slots=True)
class S1Item:
    item_id: str
    prompt: str
    answer: str
    level: str = ""
    subject: str = ""


# -- observable features -------------------------------------------------------
# TEXT ONLY. `level` and `subject` ship with the dataset and would be defensible
# as observable metadata, but including a human difficulty label in a study about
# predicting difficulty invites exactly the objection this project keeps having
# to answer. They are carried for analysis and never exposed as features.

FEATURE_NAMES = ("chars", "words_n", "digits", "latex_cmds", "has_frac",
                 "has_sqrt", "has_matrix", "has_sum", "n_equations",
                 "max_number_log10", "question_marks", "lines")

_NUM = re.compile(r"\d+")
_CMD = re.compile(r"\\[a-zA-Z]+")


def features(prompt: str) -> dict[str, float]:
    nums = [int(x) for x in _NUM.findall(prompt)[:400]]
    return {
        "chars": float(len(prompt)),
        "words_n": float(len(prompt.split())),
        "digits": float(sum(c.isdigit() for c in prompt)),
        "latex_cmds": float(len(_CMD.findall(prompt))),
        "has_frac": float("\\frac" in prompt),
        "has_sqrt": float("\\sqrt" in prompt),
        "has_matrix": float("matrix" in prompt or "\\begin{pmatrix}" in prompt),
        "has_sum": float("\\sum" in prompt or "\\prod" in prompt),
        "n_equations": float(prompt.count("$")),
        "max_number_log10": float(np.log10(max(nums) + 1)) if nums else 0.0,
        "question_marks": float(prompt.count("?")),
        "lines": float(prompt.count("\n") + 1),
    }


def feature_vector(prompt: str, names=FEATURE_NAMES) -> np.ndarray:
    f = features(prompt)
    return np.array([f[k] for k in names], float)


def grade_passthrough(item: S1Item, text: str | None) -> float:
    """Correctness is RECORDED in the dataset, not re-graded here.

    The cache stores `exact_match` in the content field as "1"/"0"; re-deriving
    it from the generation would mean reimplementing lm-eval-harness's MATH
    answer matcher and silently disagreeing with the published numbers.
    """
    return 1.0 if (text or "").strip().startswith("1") else 0.0


# -- loading -------------------------------------------------------------------

def download(benchmark: str = "math", budgets=None) -> dict[int, Path]:
    from huggingface_hub import HfApi, hf_hub_download
    want = BENCHMARKS[benchmark]
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    out = {}
    for d, b in BUDGET_DIRS.items():
        if budgets and b not in budgets:
            continue
        for f in files:
            if f.startswith(d + "/") and want in f:
                out[b] = Path(hf_hub_download(REPO, f, repo_type="dataset"))
                break
    return out


def _tokens(text: str) -> int:
    """Token count for the generation.

    Approximated at 4 characters per token rather than tokenised. The absolute
    scale does not matter -- every policy is charged on the same basis and the
    budget is expressed in the same units -- but the RATIO between budgets does,
    and that is preserved exactly by a linear approximation.
    """
    return max(1, len(text) // 4)


def _records(path: Path):
    """Yield JSON records, one per LINE FEED.

    `str.splitlines()` also splits on \x0b, \x1c, \u2028 and friends, which
    occur inside GPQA generations and produced `Unterminated string`. File
    iteration and `split("\n")` split on line feeds only, which is what JSONL
    means. The first fix here buffered until a fragment parsed, which silently
    desynchronised and yielded 4 records out of 198 -- worse than the crash,
    because it looked like success.
    """
    for line in path.read_text().split("\n"):
        if line.strip():
            yield json.loads(line)


def load(benchmark: str = "math", budgets=None) -> tuple[list[S1Item], dict]:
    """Returns (items, records) with records[budget][item_id] = outcome dict."""
    paths = download(benchmark, budgets)
    items: dict[str, S1Item] = {}
    records: dict[int, dict[str, dict]] = {}
    for b, p in sorted(paths.items()):
        rec = {}
        for r in _records(p):
            doc, did = r.get("doc", {}), str(r["doc_id"])
            iid = f"{benchmark}{int(did):05d}"
            if iid not in items:
                items[iid] = S1Item(
                    item_id=iid,
                    prompt=doc.get("problem") or doc.get("Question") or "",
                    answer=str(doc.get("answer", "")),
                    level=str(doc.get("level", "")),
                    subject=str(doc.get("subject", "")))
            resp = r.get("filtered_resps") or r.get("resps") or [""]
            while isinstance(resp, list) and resp:
                resp = resp[0]
            em = r.get("exact_match", 0)
            em = float(em[0] if isinstance(em, list) else em)
            rec[iid] = {"correct": int(round(em)), "tokens": _tokens(str(resp))}
        records[b] = rec
    return list(items.values()), records


def build_cache(path: Path, items, records, prompt_tokens: int = 64):
    """Write the external outcomes into a standard ResponseCache."""
    from governor.phase4.collect import CallRecord, ResponseCache
    cache = ResponseCache(path, model="s1-32B (simplescaling/results)")
    for b, rec in records.items():
        for it in items:
            r = rec.get(it.item_id)
            if r is None:
                continue
            used = min(r["tokens"], b)
            cache.put(it, b, CallRecord(
                it.item_id, b, "1" if r["correct"] else "0",
                "length" if used >= b else "stop",
                prompt_tokens, used, used, prompt_tokens + used, 0.0), attempts=1)
    return cache
