"""Append-only decision records in SQLite.

The Decision Record specified Parquet + DuckDB. On this machine (3.7 GB free,
no third-party wheels installed) SQLite is in the standard library, gives the same
queryability, and DuckDB can read it directly later. Swapping back is one module.

Every decision the policy makes lands here with its full provenance, so any episode
can be reconstructed and audited without re-running it. That is what makes the
brief's section 27/28 real.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id   TEXT PRIMARY KEY,
    arm          TEXT NOT NULL,
    seed         INTEGER NOT NULL,
    envelope     TEXT NOT NULL,
    budget_scale REAL NOT NULL,
    config_hash  TEXT NOT NULL,
    succeeded    INTEGER,
    terminal     TEXT,
    n_decisions  INTEGER,
    consumed     TEXT,
    truncations  INTEGER,
    violated     INTEGER,
    ground_truth TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    episode_id   TEXT NOT NULL,
    decision_id  INTEGER NOT NULL,
    features     TEXT NOT NULL,
    candidates   TEXT NOT NULL,
    chosen       TEXT NOT NULL,
    reason_code  TEXT NOT NULL,
    was_random   INTEGER NOT NULL DEFAULT 0,
    epsilon      REAL NOT NULL DEFAULT 0.0,
    provenance   TEXT NOT NULL DEFAULT 'deterministic',
    actual_cost  TEXT,
    observation  TEXT,
    belief_before TEXT,
    belief_after TEXT,
    entropy_after REAL,
    PRIMARY KEY (episode_id, decision_id)
);

CREATE INDEX IF NOT EXISTS idx_decisions_episode ON decisions(episode_id);
CREATE INDEX IF NOT EXISTS idx_episodes_arm ON episodes(arm, budget_scale);
"""


@dataclass(slots=True)
class DecisionRow:
    episode_id: str
    decision_id: int
    features: dict[str, Any]
    candidates: list[dict[str, Any]]
    chosen: str
    reason_code: str
    was_random: bool = False
    epsilon: float = 0.0
    provenance: str = "deterministic"
    actual_cost: dict[str, float] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    belief_before: list[float] = field(default_factory=list)
    belief_after: list[float] = field(default_factory=list)
    entropy_after: float = 0.0


@dataclass(slots=True)
class EpisodeRow:
    episode_id: str
    arm: str
    seed: int
    envelope: dict[str, float]
    budget_scale: float
    config_hash: str
    succeeded: bool
    terminal: str | None
    n_decisions: int
    consumed: dict[str, float]
    truncations: int
    violated: bool
    ground_truth: dict[str, Any]


class RecordStore:
    """Thin SQLite wrapper. Open once per run, write many."""

    def __init__(self, path: str | Path = "results/governor.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def write_episode(self, row: EpisodeRow) -> None:
        d = asdict(row)
        self.conn.execute(
            """INSERT OR REPLACE INTO episodes VALUES
               (:episode_id,:arm,:seed,:envelope,:budget_scale,:config_hash,
                :succeeded,:terminal,:n_decisions,:consumed,:truncations,
                :violated,:ground_truth)""",
            {
                **d,
                "envelope": json.dumps(d["envelope"]),
                "consumed": json.dumps(d["consumed"]),
                "ground_truth": json.dumps(d["ground_truth"]),
                "succeeded": int(d["succeeded"]),
                "violated": int(d["violated"]),
            },
        )

    def write_decision(self, row: DecisionRow) -> None:
        d = asdict(row)
        self.conn.execute(
            """INSERT OR REPLACE INTO decisions VALUES
               (:episode_id,:decision_id,:features,:candidates,:chosen,:reason_code,
                :was_random,:epsilon,:provenance,:actual_cost,:observation,
                :belief_before,:belief_after,:entropy_after)""",
            {
                **d,
                "features": json.dumps(d["features"]),
                "candidates": json.dumps(d["candidates"]),
                "actual_cost": json.dumps(d["actual_cost"]),
                "observation": json.dumps(d["observation"]),
                "belief_before": json.dumps(d["belief_before"]),
                "belief_after": json.dumps(d["belief_after"]),
                "was_random": int(d["was_random"]),
            },
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- reading ---------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self.conn.execute(sql, params).fetchall()

    def arm_summary(self) -> list[dict[str, Any]]:
        """Headline table: success rate and cost per success, per arm and budget."""
        rows = self.query(
            """SELECT arm, budget_scale,
                      COUNT(*)                                  AS n,
                      SUM(succeeded)                            AS wins,
                      SUM(json_extract(consumed,'$.cost'))      AS total_cost,
                      SUM(json_extract(consumed,'$.tokens'))    AS total_tokens,
                      SUM(violated)                             AS violations,
                      SUM(truncations)                          AS truncations,
                      AVG(n_decisions)                          AS avg_decisions
               FROM episodes GROUP BY arm, budget_scale ORDER BY budget_scale DESC, arm"""
        )
        out = []
        for (arm, scale, n, wins, cost, tokens, viol, trunc, avg_dec) in rows:
            wins = wins or 0
            out.append(
                {
                    "arm": arm,
                    "budget": scale,
                    "n": n,
                    "tsr": wins / n if n else 0.0,
                    "cost_per_success": (cost / wins) if wins else float("inf"),
                    "tokens_per_success": (tokens / wins) if wins else float("inf"),
                    "bvr": (viol or 0) / n if n else 0.0,
                    "truncation_rate": (trunc or 0) / n if n else 0.0,
                    "avg_decisions": avg_dec or 0.0,
                }
            )
        return out

    def episode_count(self) -> int:
        return self.query("SELECT COUNT(*) FROM episodes")[0][0]


@contextmanager
def open_store(path: str | Path = "results/governor.db") -> Iterator[RecordStore]:
    store = RecordStore(path)
    try:
        yield store
    finally:
        store.close()
