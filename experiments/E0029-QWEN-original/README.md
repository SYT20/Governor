# E0029-QWEN-original — ENGINEERING-INVALID

The first analysis of the Qwen run. **Its scientific conclusion does not stand**,
because the feature vector it tested was not the one it was designed to test.

## What happened

`decision_features()` read the eleven static/AST features as precomputed row
columns, defaulting to `0.0`. E0028's rows carried those columns, so it worked
there. E0029's generation rows carry `code` and nothing else, so all eleven were
zero for all 4750 samples. Nothing raised. Nothing warned.

The run therefore tested five distinct observables, not twenty-five.

## Why it is kept

A withdrawn result that disappears is worse than one that stays with its reason
attached. This directory is the reason, and it is referenced by
`E0029-QWEN-corrected`.

## What survives

The expensive stages are unaffected and were NOT repeated:

* 4750 generation rows
* 4750 private-test grades — 65,280 executions, 91.4 min, 0 harness failures
* the ceiling, **+0.1378**, computed from the hidden label alone

## What does not

The gate-3 figure of **0.415** and any reading of it as "observable features
carry no signal". With the static features restored the same data gives 0.557
for the 25-feature model and 0.651 for `code_lines` alone.
