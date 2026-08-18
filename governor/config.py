"""One place to resolve provider API keys, with an error you can act on.

Three things made key setup harder than it needed to be:

  1. The variable names disagreed with each other -- `OR_KEY`, `Groq`,
     `GEMINI_KEY`. Two of the three are not what the provider's own docs call
     them, and one is mixed case, which is unusual enough that people set
     `GROQ` and wonder why nothing happens.
  2. `.env` was looked for relative to the CURRENT directory, so the harness
     worked from the repo root and failed from anywhere else with a message
     that read like the key was missing.
  3. The failure said what was absent but not what to do about it.

So: canonical names that match each provider's documentation, the old names
kept working as aliases, `.env` found from the repository root wherever you
run from, and a failure that prints the three ways to fix it.

Nothing here ever reads a key from source, and nothing prints one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class KeySpec:
    """Where one provider's credential comes from and what it looks like."""

    provider: str
    canonical: str                  # the name we document
    aliases: tuple[str, ...]        # older names, still honoured
    signup: str                     # where to get one
    prefix: str = ""                # expected prefix, for a sanity check only

    @property
    def names(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)


KEY_SPECS: dict[str, KeySpec] = {
    "openrouter": KeySpec(
        provider="openrouter",
        canonical="OPENROUTER_API_KEY",
        aliases=("OR_KEY",),
        signup="https://openrouter.ai/keys",
        prefix="sk-or-",
    ),
    "groq": KeySpec(
        provider="groq",
        canonical="GROQ_API_KEY",
        aliases=("Groq", "GROQ"),
        signup="https://console.groq.com/keys",
        prefix="gsk_",
    ),
    "gemini": KeySpec(
        provider="gemini",
        canonical="GEMINI_API_KEY",
        aliases=("GEMINI_KEY",),
        signup="https://aistudio.google.com/apikey",
        prefix="AIza",
    ),
}


def repo_root() -> Path:
    """The repository root, so `.env` resolves the same from any directory."""
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    """Parse the untracked `.env` at the repository root.

    Deliberately tiny: `KEY=value`, `export KEY=value`, `#` comments, optional
    surrounding quotes. No dependency, no interpolation, no surprises.
    """
    out: dict[str, str] = {}
    path = repo_root() / ".env"
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, _, value = line.partition("=")
        out[name.strip()] = value.strip().strip("'\"")
    return out


class MissingKey(RuntimeError):
    """Raised with instructions, not just a diagnosis."""


def _spec(provider: str) -> KeySpec:
    try:
        return KEY_SPECS[provider.lower()]
    except KeyError:
        known = ", ".join(sorted(KEY_SPECS))
        raise MissingKey(f"unknown provider {provider!r}; known: {known}") from None


def resolve_key(provider: str, *, required: bool = True) -> str:
    """Return the API key for `provider`, or raise telling the caller how to set one.

    Search order, first hit wins: the canonical environment variable, then its
    aliases, then the same names in `.env` at the repository root. The
    environment beats `.env` so a shell override always works.
    """
    spec = _spec(provider)

    for name in spec.names:
        if (v := os.environ.get(name, "").strip()):
            return v

    env_file = _dotenv()
    for name in spec.names:
        if (v := env_file.get(name, "").strip()):
            return v

    if not required:
        return ""

    raise MissingKey(
        f"\nNo API key found for {spec.provider}.\n\n"
        f"Any one of these works:\n\n"
        f"  1. Put it in .env at the repo root   (easiest, and gitignored)\n"
        f"       echo '{spec.canonical}=your-key-here' >> {repo_root()/'.env'}\n\n"
        f"  2. Export it for this shell\n"
        f"       export {spec.canonical}=your-key-here\n\n"
        f"  3. Set it in your MCP client config, so the harness inherits it\n"
        f"       \"env\": {{\"{spec.canonical}\": \"your-key-here\"}}\n\n"
        f"Get a key: {spec.signup}\n"
        f"Check what the harness can see: python -m governor.config\n"
    )


def key_status() -> list[dict[str, object]]:
    """Report which keys are visible and where each came from. Never returns a key."""
    env_file = _dotenv()
    rows: list[dict[str, object]] = []
    for spec in KEY_SPECS.values():
        source, name_used, value = "", "", ""
        for name in spec.names:
            if (v := os.environ.get(name, "").strip()):
                source, name_used, value = "environment", name, v
                break
        if not source:
            for name in spec.names:
                if (v := env_file.get(name, "").strip()):
                    source, name_used, value = ".env", name, v
                    break
        rows.append({
            "provider": spec.provider,
            "found": bool(source),
            "source": source,
            "variable": name_used,
            "canonical": spec.canonical,
            "deprecated_name": bool(name_used) and name_used != spec.canonical,
            "prefix_ok": (not value) or (not spec.prefix) or value.startswith(spec.prefix),
            "signup": spec.signup,
        })
    return rows


def main() -> int:
    """`python -m governor.config` -- show what the harness can see."""
    rows = key_status()
    width = max(len(str(r["provider"])) for r in rows)
    print("\nAPI keys visible to the harness\n")
    for r in rows:
        if r["found"]:
            mark, detail = "  set  ", f"{r['variable']} from {r['source']}"
            if r["deprecated_name"]:
                detail += f"  (deprecated name -- prefer {r['canonical']})"
            if not r["prefix_ok"]:
                detail += "  (WARNING: unexpected prefix, is this the right key?)"
        else:
            mark, detail = "not set", f"set {r['canonical']}  --  {r['signup']}"
        print(f"  [{mark}]  {str(r['provider']):<{width}}   {detail}")

    missing = [r for r in rows if not r["found"]]
    print(f"\n  {len(rows) - len(missing)}/{len(rows)} providers configured.")
    print("  No key is needed for `make test`, `make verify` or `make smoke`.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
