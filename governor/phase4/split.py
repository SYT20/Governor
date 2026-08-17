"""The frozen selection / evaluation split, by ITEM ID.

WHY BY ID AND NOT BY SLICE. `pool[:40]` depends on which items happen to be
cached, so the split silently moves as collection proceeds. Freezing the actual
item ids to a file removes that: the split is a fact on disk that predates the
selection, and `verify_disjoint` can check it later from the ids alone.

WHY IT EXISTS. A reviewer flagged that the Phase 4R structural search might have
selected its configuration using items later called held-out. Checking the
record, the SELECTED configuration's S1/S2 came from 40 items, not 89 -- only the
rejected (700, 2800) pairs saw the larger pool. So the winner's qualification was
clean. But the comparison that rejected the alternatives was not, and a protocol
that requires reading a metrics file to establish that is not a protocol. The
split is now declared before the search and enforced by a trap check.

SELECTION items are the first 40 of the calibration pool -- the curve items,
which is what S1 and S2 were computed on. Everything after is EVALUATION and
must never touch configuration, budget, mode or hyperparameter choice.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from governor.phase4.tasks import make_pool

SPLIT_FILE = Path("configs/phase4r_split.json")
N_SELECTION = 40
POOL_SEED, POOL_N = 1000, 400


def build() -> dict:
    """Deterministic from the pool seed alone -- not from cache state."""
    pool = make_pool(POOL_SEED, POOL_N)
    sel = [i.item_id for i in pool[:N_SELECTION]]
    evl = [i.item_id for i in pool[N_SELECTION:]]
    payload = {"pool_seed": POOL_SEED, "pool_n": POOL_N,
               "n_selection": N_SELECTION,
               "selection_ids": sel, "evaluation_ids": evl}
    payload["sha256"] = hashlib.sha256(
        json.dumps({k: payload[k] for k in sorted(payload)},
                   sort_keys=True).encode()).hexdigest()
    return payload


def freeze(path: Path = SPLIT_FILE) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return load(path)
    payload = build()
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load(path: Path = SPLIT_FILE) -> dict:
    return json.loads(path.read_text())


def selection_ids(path: Path = SPLIT_FILE) -> set[str]:
    return set(freeze(path)["selection_ids"])


def evaluation_ids(path: Path = SPLIT_FILE) -> set[str]:
    return set(freeze(path)["evaluation_ids"])


def filter_selection(items, path: Path = SPLIT_FILE):
    ids = selection_ids(path)
    return [i for i in items if i.item_id in ids]


def filter_evaluation(items, path: Path = SPLIT_FILE):
    ids = evaluation_ids(path)
    return [i for i in items if i.item_id in ids]


def verify_disjoint(path: Path = SPLIT_FILE) -> tuple[bool, str]:
    d = freeze(path)
    s, e = set(d["selection_ids"]), set(d["evaluation_ids"])
    both = s & e
    fresh = build()
    drifted = fresh["sha256"] != d["sha256"]
    if both:
        return False, f"{len(both)} ids in BOTH halves: {sorted(both)[:5]}"
    if drifted:
        return False, "the split on disk no longer matches the pool it names"
    return True, f"selection={len(s)} evaluation={len(e)} disjoint, sha ok"
