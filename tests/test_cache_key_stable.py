"""The cache key is a FROZEN format.

Changing it invalidates every collected response. That happened once: folding
the system prompt into the key turned 262 already-paid-for rows into misses,
and the next run would have re-collected them against a hard cap of 1000
requests per day. The cost of a silent key change is a day of quota, so the
format is pinned here.
"""
from governor.phase4.collect import _key


def test_key_format_is_frozen():
    k = _key("m/model", "i000001", "Compute: 2 * 3", 700, 0.0)
    assert k == "i000001:700:8cce741f31f7bd97", k


def test_key_ignores_the_system_prompt():
    """Families are separated by cache file, item-id prefix and prompt text."""
    import inspect
    sig = inspect.signature(_key)
    assert list(sig.parameters) == ["model", "item_id", "prompt", "max_tokens",
                                    "temp"]


def test_key_varies_with_everything_it_should():
    base = dict(model="m", item_id="i1", prompt="p", max_tokens=700, temp=0.0)
    k0 = _key(**base)
    for field, other in (("model", "n"), ("item_id", "i2"), ("prompt", "q"),
                         ("max_tokens", 2800), ("temp", 0.5)):
        assert _key(**{**base, field: other}) != k0, field
