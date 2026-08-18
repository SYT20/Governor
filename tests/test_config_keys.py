"""Key resolution must be predictable, backward-compatible, and never leak.

The harness previously used three disagreeing variable names and looked for
`.env` relative to the current directory, so it worked from the repo root and
failed anywhere else with a message that read like a missing key.
"""
from __future__ import annotations

import pytest

from governor.config import (
    KEY_SPECS, MissingKey, key_status, repo_root, resolve_key,
)


def test_environment_beats_dotenv(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-env")
    assert resolve_key("openrouter") == "sk-or-from-env"


def test_deprecated_names_still_work(monkeypatch):
    """Existing setups must not break when the canonical name changes."""
    for provider, old in (("openrouter", "OR_KEY"),
                          ("groq", "Groq"),
                          ("gemini", "GEMINI_KEY")):
        for name in KEY_SPECS[provider].names:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(old, f"value-via-{old}")
        assert resolve_key(provider) == f"value-via-{old}"


def test_canonical_wins_over_alias(monkeypatch):
    monkeypatch.setenv("OR_KEY", "old")
    monkeypatch.setenv("OPENROUTER_API_KEY", "new")
    assert resolve_key("openrouter") == "new"


def test_missing_key_explains_how_to_fix_it(monkeypatch):
    for name in KEY_SPECS["gemini"].names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("governor.config._dotenv", lambda: {})

    with pytest.raises(MissingKey) as excinfo:
        resolve_key("gemini")

    msg = str(excinfo.value)
    assert "GEMINI_API_KEY" in msg          # the name to set
    assert "aistudio.google.com" in msg     # where to get one
    assert ".env" in msg and "export" in msg and "env" in msg   # all three routes


def test_not_required_returns_empty_instead_of_raising(monkeypatch):
    for name in KEY_SPECS["gemini"].names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("governor.config._dotenv", lambda: {})
    assert resolve_key("gemini", required=False) == ""


def test_unknown_provider_names_the_known_ones():
    with pytest.raises(MissingKey, match="openrouter"):
        resolve_key("not-a-provider")


def test_dotenv_is_found_from_repo_root_not_cwd(tmp_path, monkeypatch):
    """Running from a subdirectory must not change which .env is used."""
    monkeypatch.chdir(tmp_path)
    assert repo_root().name == "Atlan Proj" or (repo_root() / "governor").is_dir()


def test_status_never_returns_a_key_value(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-super-secret-value")
    blob = repr(key_status())
    assert "super-secret" not in blob
    assert "sk-or-super" not in blob


def test_status_flags_a_wrong_looking_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-or-this-is-an-openrouter-key")
    row = next(r for r in key_status() if r["provider"] == "groq")
    assert row["found"] is True
    assert row["prefix_ok"] is False        # gsk_ expected, caught before a 401


def test_every_spec_documents_where_to_get_a_key():
    for spec in KEY_SPECS.values():
        assert spec.signup.startswith("https://")
        assert spec.canonical.isupper()
