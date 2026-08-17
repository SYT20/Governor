"""Response cache + collector for the Phase 4 LLM arm.

TWO PROBLEMS THIS SOLVES, BOTH LEARNED THE HARD WAY.

1. THROTTLING VOIDS RUNS. The Gemini curve returned 492/500 HTTP 429 and the
   script printed a curve anyway, because a failed call and a wrong answer were
   the same row. Here a throttled call is NEVER cached. It is retried with
   backoff, and if the throttle is sustained the collector RAISES -- a partial
   cache is resumable, a cache full of silent failures is not.

2. SEVEN POLICIES CANNOT EACH PAY FOR THE SAME CALL. Env 6 compared policies on
   shared random draws (`roll`) so the difference between them was the decision
   and not the noise. The same idea applies here: every (item, budget) pair is
   called ONCE and every policy reads the same recorded response. At ~10 s and a
   free-tier daily cap per call, this is also the difference between a run that
   is possible and one that is not.

   This is common random numbers, not a shortcut. Temperature is 0, so the
   responses are near-deterministic anyway; caching makes the comparison exactly
   matched rather than approximately so, and it is applied identically to every
   policy including the oracle.

The cache is the experimental record: it stores the reply, the parsed answer,
correctness, and the token counts that become the CHARGED COST.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from governor.phase4.tasks import SYSTEM_PROMPT, Item, is_correct, parse_answer

@dataclass(frozen=True, slots=True)
class Provider:
    """A chat-completions endpoint. Two are wired: OpenRouter and Groq.

    Groq returns Cloudflare 403 (error 1010) to the default `Python-urllib`
    User-Agent, which looks exactly like an auth failure and is not one. Every
    request therefore carries an explicit UA.
    """
    name: str
    url: str
    key_env: str
    budget_field: str          # the parameter that actually caps generation


OPENROUTER = Provider("openrouter", "https://openrouter.ai/api/v1/chat/completions",
                      "OR_KEY", "max_tokens")
GROQ = Provider("groq", "https://api.groq.com/openai/v1/chat/completions",
                "Groq", "max_completion_tokens")
PROVIDERS = {p.name: p for p in (OPENROUTER, GROQ)}

MODEL = "nvidia/nemotron-nano-9b-v2:free"
CACHE = Path("results/p4_cache.sqlite")
UA = "governor-research/1.0"

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key TEXT PRIMARY KEY,
    model TEXT, item_id TEXT, max_tokens INTEGER, temperature REAL,
    prompt_sha TEXT,
    content TEXT, finish_reason TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    reasoning_tokens INTEGER, total_tokens INTEGER,
    latency_s REAL, fetched_utc TEXT, attempts INTEGER
);
CREATE INDEX IF NOT EXISTS idx_item ON calls(item_id, max_tokens);
"""


def _key(model: str, item_id: str, prompt: str, max_tokens: int, temp: float) -> str:
    h = hashlib.sha256(f"{model}|{prompt}|{max_tokens}|{temp}".encode()).hexdigest()
    return f"{item_id}:{max_tokens}:{h[:16]}"


@dataclass(slots=True)
class CallRecord:
    """One measured LLM call. `total_tokens` is what the environment charges."""
    item_id: str
    max_tokens: int
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    latency_s: float

    @property
    def answered(self) -> bool:
        return bool(self.content.strip())

    @property
    def starved(self) -> bool:
        return self.finish_reason == "length"


class RateLimited(RuntimeError):
    """Sustained throttling. Raised instead of recording a degraded run."""


class TokenPacer:
    """Groq bills TPM against RESERVED `max_completion_tokens`, not actual use.

    Measured: two calls with caps 300 and 1400 and ~112-token prompts drew 1936
    from the 8000/min bucket while emitting 420 tokens. A collector that paces
    on observed usage would therefore be throttled at roughly a quarter of the
    rate it thinks it is running, which is how the Gemini curve died. Pace on
    what is reserved.
    """

    def __init__(self, tpm: int, safety: float = 0.85):
        self.capacity = tpm * safety
        self.tokens = self.capacity
        self.rate = self.capacity / 60.0
        self.t = time.time()
        self.lock = threading.Lock()

    def acquire(self, n: int) -> None:
        n = min(n, self.capacity)
        while True:
            with self.lock:
                now = time.time()
                self.tokens = min(self.capacity,
                                  self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            time.sleep(min(wait, 5.0))


class ResponseCache:
    def __init__(self, path: Path = CACHE, model: str = MODEL,
                 temperature: float = 0.0, provider: Provider = OPENROUTER):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model, self.temperature, self.provider = model, temperature, provider
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.lock = threading.Lock()

    # -- reads ---------------------------------------------------------------

    def get(self, item: Item, max_tokens: int) -> CallRecord | None:
        k = _key(self.model, item.item_id, item.prompt, max_tokens, self.temperature)
        with self.lock:
            row = self.conn.execute(
                "SELECT content,finish_reason,prompt_tokens,completion_tokens,"
                "reasoning_tokens,total_tokens,latency_s FROM calls WHERE key=?",
                (k,)).fetchone()
        if row is None:
            return None
        return CallRecord(item.item_id, max_tokens, row[0], row[1], row[2],
                          row[3], row[4], row[5], row[6])

    def require(self, item: Item, max_tokens: int) -> CallRecord:
        r = self.get(item, max_tokens)
        if r is None:
            raise KeyError(f"no cached call for {item.item_id} @ {max_tokens}. "
                           f"The environment must never call the API mid-episode: "
                           f"collect first, then execute.")
        return r

    def count(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def missing(self, items: Iterable[Item], budgets: Iterable[int]) -> list[tuple[Item, int]]:
        return [(it, b) for it in items for b in budgets if self.get(it, b) is None]

    # -- writes --------------------------------------------------------------

    def put(self, item: Item, max_tokens: int, rec: CallRecord, attempts: int) -> None:
        k = _key(self.model, item.item_id, item.prompt, max_tokens, self.temperature)
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (k, self.model, item.item_id, max_tokens, self.temperature,
                 hashlib.sha256(item.prompt.encode()).hexdigest()[:16],
                 rec.content, rec.finish_reason, rec.prompt_tokens,
                 rec.completion_tokens, rec.reasoning_tokens, rec.total_tokens,
                 rec.latency_s,
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), attempts))
            self.conn.commit()


# -- fetching ------------------------------------------------------------------

def _one_call(key: str, provider: Provider, model: str, item: Item,
              max_tokens: int, temperature: float,
              timeout: int = 180) -> CallRecord:
    body = {"model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": item.prompt}],
            provider.budget_field: int(max_tokens), "temperature": temperature}
    t0 = time.time()
    req = urllib.request.Request(
        provider.url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    if "error" in d:                       # 200 with an error body: still a failure
        code = str((d["error"] or {}).get("code", ""))
        msg = str(d["error"])[:200]
        if "429" in code or "rate" in msg.lower():
            raise urllib.error.HTTPError(provider.url, 429, msg, None, None)
        raise RuntimeError(msg)
    ch = d["choices"][0]
    msg_o = ch["message"]
    u = d.get("usage", {}) or {}
    # Two conventions for where the reasoning goes: a separate `reasoning`
    # field (OpenRouter/nemotron, Groq/gpt-oss) or inline <think> in content
    # (Groq/qwen). Concatenating keeps `content` a faithful record of what the
    # model emitted, which is what the grader must see.
    content = (msg_o.get("content") or "").strip()
    sep = (msg_o.get("reasoning") or "").strip()
    if sep and "<think>" not in content:
        content = f"<think>\n{sep}\n</think>\n{content}"
    return CallRecord(
        item_id=item.item_id, max_tokens=int(max_tokens),
        content=content,
        finish_reason=str(ch.get("finish_reason") or ""),
        prompt_tokens=int(u.get("prompt_tokens", 0)),
        completion_tokens=int(u.get("completion_tokens", 0)),
        reasoning_tokens=int((u.get("completion_tokens_details") or {})
                             .get("reasoning_tokens", 0)),
        total_tokens=int(u.get("total_tokens", 0)),
        latency_s=time.time() - t0)


def collect(cache: ResponseCache, items: list[Item], budgets: list[int],
            api_key: str, workers: int = 4, max_retries: int = 4,
            throttle_abort: int = 25, progress_every: int = 25,
            deadline_s: float | None = None,
            tpm: int | None = None) -> dict[str, int]:
    """Fill the cache for every (item, budget) not already present.

    Resumable: only missing pairs are fetched, so an interrupted run costs
    nothing. `throttle_abort` consecutive 429s raises RateLimited rather than
    letting the run degrade into the Gemini failure.
    """
    todo = cache.missing(items, budgets)
    if not todo:
        return {"fetched": 0, "cached": len(items) * len(budgets), "errors": 0}

    q: queue.Queue = queue.Queue()
    for pair in todo:
        q.put(pair)
    state = {"fetched": 0, "errors": 0, "consec_429": 0, "stop": False,
             "t0": time.time()}
    slock = threading.Lock()
    errors: list[str] = []
    pacer = TokenPacer(tpm) if tpm else None
    PROMPT_RESERVE = 160          # upper bound on prompt tokens, checked below

    def worker() -> None:
        while not state["stop"]:
            try:
                item, b = q.get_nowait()
            except queue.Empty:
                return
            for attempt in range(1, max_retries + 1):
                if state["stop"]:
                    return
                try:
                    if pacer:
                        pacer.acquire(b + PROMPT_RESERVE)
                    rec = _one_call(api_key, cache.provider, cache.model, item,
                                    b, cache.temperature)
                    cache.put(item, b, rec, attempt)
                    with slock:
                        state["fetched"] += 1
                        state["consec_429"] = 0
                        n = state["fetched"]
                        if n % progress_every == 0:
                            el = time.time() - state["t0"]
                            print(f"    {n}/{len(todo)} fetched  "
                                  f"{el/60:.1f} min  {el/max(n,1):.1f} s/call  "
                                  f"eta {(len(todo)-n)*el/max(n,1)/60:.0f} min",
                                  flush=True)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        with slock:
                            state["consec_429"] += 1
                            if state["consec_429"] >= throttle_abort:
                                state["stop"] = True
                        time.sleep(min(60, 4 * 2 ** attempt))
                        continue
                    if attempt == max_retries:
                        with slock:
                            state["errors"] += 1
                            errors.append(f"{item.item_id}@{b}: HTTP {e.code}")
                    time.sleep(2 * attempt)
                except Exception as e:                       # noqa: BLE001
                    if attempt == max_retries:
                        with slock:
                            state["errors"] += 1
                            errors.append(f"{item.item_id}@{b}: {str(e)[:90]}")
                    time.sleep(2 * attempt)
            if deadline_s and time.time() - state["t0"] > deadline_s:
                state["stop"] = True

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    if state["consec_429"] >= throttle_abort:
        raise RateLimited(
            f"{throttle_abort} consecutive HTTP 429 after {state['fetched']} "
            f"calls. Cache retained ({cache.count()} rows) -- rerun to resume. "
            f"NOT recording a throttled run as data.")
    return {"fetched": state["fetched"], "errors": state["errors"],
            "cached": cache.count(), "error_sample": errors[:5]}


def outcome(cache: ResponseCache, item: Item, max_tokens: int) -> dict:
    """The full measured outcome of one (item, budget) call, for the env."""
    r = cache.require(item, max_tokens)
    return {"correct": int(is_correct(item, r.content)),
            "answered": int(r.answered), "starved": int(r.starved),
            "parsed": parse_answer(r.content),
            "total_tokens": r.total_tokens,
            "completion_tokens": r.completion_tokens,
            "reasoning_tokens": r.reasoning_tokens,
            "prompt_tokens": r.prompt_tokens,
            "latency_s": r.latency_s, "finish_reason": r.finish_reason}


def api_key(provider: Provider = OPENROUTER) -> str:
    """Keys come from the environment or the untracked .env, never from source.

    `.env` is gitignored and `tests/test_no_secrets.py` scans the tree; a key
    was committed once in this project and the fix has to be structural.
    """
    k = os.environ.get(provider.key_env, "")
    if not k:
        env = Path(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith(f"{provider.key_env}="):
                    k = line.split("=", 1)[1].strip().strip("'\"")
    if not k:
        raise RuntimeError(
            f"{provider.key_env} is not set (checked environment and ./.env).")
    return k
