# Aborted runs — retained as evidence, never as data

## `e0029-gptoss-ABORTED.jsonl` — 76 samples, `openai/gpt-oss-120b`, 2026-08-20

Stopped after 76 of 4750 samples. **These rows must never enter an analysis.**

### Why it was stopped

A `max_completion_tokens` cap of 2500 was manufacturing failures:

| | samples | empty extracted code |
|---|---|---|
| at the 2500 cap | 26 | 11 (42%) |
| under the cap | 50 | 0 (0%) |

Perfect separation. `gpt-oss-120b` is a reasoning model and spends most of its
budget reasoning before emitting code, so a third of samples were cut off
mid-thought and produced nothing. Those would have been scored as the model
failing the problem when the configuration failed instead.

The confound points straight at the quantity being measured: harder problems
reason longer, so the artificial failure rate would rise with difficulty — the
exact signal the allocation experiment is trying to detect.

### The general lesson, now enforced

**The token budget is part of the experimental instrument.** Changing
`max_completion_tokens` changes what the model does, and therefore changes the
experiment. `governor/execfeedback/preflight.py` now requires a truncation and
empty-output check before any full generation run, and the harness refuses a run
whose preflight fails.

### Also recorded: the cost estimate was wrong twice

| estimate | tokens/sample | basis |
|---|---|---|
| first | 388 | toy prompt |
| second | 1829 | 3 real problems |
| actual | 2378 | 76 real samples |

11.3M tokens total, a 23.5-hour floor at the free tier's 8000 tokens/min, with
31–36 hours observed. Raising the cap to stop the truncation would have pushed it
past 40 hours.
