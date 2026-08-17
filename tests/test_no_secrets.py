"""Secret scan. A credential was committed into scripts/llm_m2_curve.py during
the LLMM2 work; this makes that class of mistake fail the suite instead of
reaching a remote.

It scans the WORKING TREE only. Purging git history requires a rewrite and is
recorded as a manual action -- the exposed key must be rotated regardless,
since it has already left the machine.
"""
from __future__ import annotations

import pathlib
import re

# split so this file does not itself trip the scan
PATTERNS = [re.compile(p) for p in (
    r"sk-" + r"or-v1-[A-Za-z0-9]{16,}",
    r"sk-" + r"ant-[A-Za-z0-9\-]{16,}",
    r"AKIA[0-9A-Z]{16}",
    r"ghp_[A-Za-z0-9]{20,}",
)]
ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", ".pytest_cache", "node_modules"}


def test_no_credentials_in_tracked_sources():
    hits = []
    for f in ROOT.rglob("*"):
        if not f.is_file() or any(p in SKIP for p in f.parts):
            continue
        if f.suffix not in {".py", ".md", ".json", ".txt", ".sh", ".yaml", ".yml"}:
            continue
        if f.name == pathlib.Path(__file__).name:
            continue
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for pat in PATTERNS:
            if pat.search(text):
                hits.append(f"{f.relative_to(ROOT)}: {pat.pattern}")
    assert not hits, "credentials found in working tree:\n  " + "\n  ".join(hits)


def test_llm_adapter_reads_key_from_environment_only():
    src = (ROOT / "governor" / "gate" / "llm_m2.py").read_text()
    assert 'os.environ.get("OR_KEY"' in src
    assert "sk-" + "or-v1" not in src
