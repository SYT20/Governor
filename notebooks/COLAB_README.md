# Governor on Colab — what it is and what it is not

**The notebook is an execution layer around the existing harness, not a second
implementation of it.**

```
                 SAME GOVERNOR PROJECT
                         │
              ┌──────────┴──────────┐
        Local execution        Colab execution
              │                     │
         Local M2               Qwen M2
              └──────────┬──────────┘
                  Same contracts
                  Same evaluation
                  Same guardrails
                  Same ledger
```

Every operation resolves to a repository source file, a pinned dependency, an
explicit config, or a raw artifact — never to "a function defined five cells
ago". That is why the notebook is thin: `scripts/colab_*.py` hold the logic, and
the cells orchestrate them. A Colab-only code path could quietly diverge from the
harness it is supposed to be testing, and then the experiment would measure the
notebook rather than the architecture.

## Files

| File | Role |
|---|---|
| `notebooks/governor_colab_e0029_qwen.ipynb` | orchestration, 19 cells, sections 00–19 |
| `scripts/colab_bootstrap.py` | clone, pin, install, diagnose → `COLAB_BOOTSTRAP = PASS` |
| `scripts/colab_text_loader.py` | load modules from source text; compare against normal import |
| `scripts/colab_preflight.py` | nine gates → `READY FOR FULL RUN` or the failing gate |
| `scripts/verify_colab_run.py` | **independent** recomputation from raw artifacts |
| `configs/colab-requirements.txt` | pinned dependencies, upper bounds included |
| `configs/colab_model.json` | backend config; no cell hard-codes a model |

## The rule that matters most

**Colab is a worker, not the source of truth.**

```
COLAB RESULT → artifacts → Drive → repository → Claude Code recomputes
             → guardrails re-run → reproducibility re-checked → only then promoted
```

`verify_colab_run.py` ignores every headline figure in the handoff and rebuilds it
from `raw/*.jsonl`. It has been tested against four kinds of tampering and
catches all of them:

| Tampering | Caught by |
|---|---|
| summary claims a better utility than the rows support | numerical recomputation |
| a forbidden field added to raw rows | information-boundary check, independently of checksums |
| calibration and evaluation made to overlap | split disjointness |
| raw rows edited after the fact | SHA256 |

On a disagreement the status is `VERIFICATION_FAILED` and **the summary is never
edited to match**. The discrepancy is the finding.

## Why each gate exists

Not one of these is hypothetical. Each corresponds to a run that was abandoned.

**Truncation guard.** `gpt-oss-120b` at a 2500-token cap: 34% of samples
truncated and **42% of those produced no code**, against 0% of uncapped samples.
Because harder problems reason longer, the artificial failure rate would have
risen with difficulty — precisely the signal the experiment tries to measure.
The generation ceiling is now chosen from a measured length distribution
(p50/p90/p95/max) rather than picked.

**Thinking-mode check.** Qwen3 with default thinking enabled consumed the whole
budget without emitting code on 3/3 sampled problems, projecting 191 hours.
Disabling it: median 212 tokens, 2% truncation, 0% empty.

**Import parity.** Notebook state hides staleness. Loading each module both ways
and requiring agreement catches a cache pointing at code that no longer exists.
Building this found a real bug in the loader itself: `dataclasses` resolves
`sys.modules[cls.__module__].__dict__`, so a module absent from `sys.modules`
fails every decorated class.

**Information boundary.** The gate probes six forbidden field names and requires
rejection. On first run it rejected **two of six** — `oracle_leakage`'s list was
written for the earlier task families and had no term for a hidden test, a
judge's verdict, or a reference solution. A feature called `graded` would have
passed silently. The list is now extended by eleven terms.

**Sandbox containment.** Generated code is untrusted and never runs in the
notebook process: subprocess, process-group kill, hard timeout, `RLIMIT_AS`,
`RLIMIT_NPROC`, `RLIMIT_FSIZE`, output caps. Eight adversarial cases including
fork bomb, memory bomb and stdout flood.

**Colab limitation, recorded honestly:** where a container cannot enforce a
resource limit, the limitation is recorded rather than a stronger guarantee
being claimed.

## Cost estimates have been wrong four times

Recorded because the pattern matters more than any single number:

| Estimate | Value | Basis |
|---|---|---|
| tokens/sample, 1st | 388 | toy prompt |
| tokens/sample, 2nd | 1829 | 3 real problems |
| tokens/sample, actual | 2378 | 76 real samples |
| runtime, from 3 samples | 13 h | small sample |
| runtime, from 100 samples | 21.9 h | preflight |

Small samples of latency have been systematically optimistic here. The preflight
projects from at least 100 samples for that reason.

## Verdicts

`PASS` · `INCONCLUSIVE` · `BLOCKED` · `SAMPLE-UNATTAINABLE` · `ENGINEERING-FAILURE`

`VERIFIED` is never emitted merely because the notebook completed. Engineering
and science are separate axes: a clean run whose CI crosses zero is
`ENGINEERING_VERIFIED` with `SCIENTIFIC_STATUS = INCONCLUSIVE`.
