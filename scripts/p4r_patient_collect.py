#!/usr/bin/env python3
"""Collect LOW=300 responses as the per-day token bucket refills.

Groq's TPD refills continuously at roughly 10k tokens/hour, so the 49 remaining
one-call items arrive over about two hours rather than all at once. This waits
instead of hammering: a TPD 429 means come back later, and burning the retry
ladder on it just hides the message.

Resumable and idempotent -- it only ever fetches what the cache is missing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.phase4.collect import (  # noqa: E402
    DailyQuotaExhausted, RateLimited, ResponseCache, api_key, collect,
)
from governor.phase4.config import CAL_POOL_SEED, ENGINES  # noqa: E402
from governor.phase4.split import filter_evaluation, freeze  # noqa: E402
from governor.phase4.tasks import make_pool  # noqa: E402

WAIT_S = 900

def main() -> int:
    cfg = ENGINES["qwen"]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    pool = make_pool(CAL_POOL_SEED, 400)
    key = api_key(cfg["provider"])
    for attempt in range(40):
        # EVALUATION IDS ONLY, by construction. The previous version selected
        # "has 700, lacks 300", which happened to be the evaluation half only
        # because every selection item already had 300 -- incidental, not
        # enforced. Scarce quota must not be spent enlarging the set a
        # configuration was chosen on.
        ev = filter_evaluation(pool)
        need = [i for i in ev if cache.get(i, 700) and not cache.get(i, 300)]
        have = sum(1 for i in ev if cache.get(i, 300) and cache.get(i, 700))
        print(f"[{time.strftime('%H:%M:%S')}] {have}/24 EVALUATION items ready, "
              f"{len(need)} still need LOW=300", flush=True)
        if have >= 24 and not need:
            print("ENOUGH FOR THE GATE", flush=True)
        if not need:
            print("COLLECTION COMPLETE", flush=True)
            return 0
        try:
            st = collect(cache, need, [300], key, workers=3,
                         tpm=cfg["tpm"], progress_every=10)
            print(f"  {st}", flush=True)
        except (DailyQuotaExhausted, RateLimited) as e:
            print(f"  quota gone ({str(e)[:80]}); sleeping {WAIT_S}s", flush=True)
            time.sleep(WAIT_S)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
