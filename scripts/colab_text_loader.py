#!/usr/bin/env python3
"""Load repository modules from source text, with no reliance on notebook state.

WHY THIS EXISTS

A Colab notebook accumulates hidden state: functions defined five cells ago,
`sys.path` entries added by a cell that has since been edited, `importlib` caches
holding a module that no longer matches the file on disk. Code that works in a
long-lived session can fail completely after `Runtime -> Restart session`, and
the failure looks like a code bug rather than a state bug.

This loader removes that class of failure by construction. It reads a file as
TEXT, compiles it, and executes it in a namespace it owns. Nothing it returns
depends on what any earlier cell did.

It is deliberately NOT a replacement for normal imports. The point is to have a
second, independent path so the two can be COMPARED -- `verify_import_parity`
loads the same module both ways and fails if they disagree. A notebook that
passes both has proved it can recover from a fresh kernel; one that passes only
the normal path may simply be reading stale state.

USAGE

    from scripts.colab_text_loader import load_source_module, repo_root

    ledger = load_source_module(repo_root() / "governor/harness/ledger.py",
                                "governor.harness.ledger")

For a module with intra-package imports, use `load_package_module`, which puts
the repository on `sys.path` for the duration of the load and then restores it.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
import types
from contextlib import contextmanager
from pathlib import Path


class LoaderError(RuntimeError):
    """Raised with the reason, never a bare ImportError from deep in the stack."""


def repo_root(start: Path | None = None) -> Path:
    """Find the repository root by marker file, not by assuming a cwd."""
    here = (start or Path(__file__)).resolve()
    for cand in [here, *here.parents]:
        if (cand / "governor").is_dir() and (cand / "Makefile").exists():
            return cand
    raise LoaderError(
        f"repository root not found above {here}. Expected a directory containing "
        "both governor/ and Makefile.")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _repo_on_path(root: Path):
    added = str(root) not in sys.path
    if added:
        sys.path.insert(0, str(root))
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass


def load_source_module(path: str | Path, module_name: str,
                       *, allow_package_imports: bool = False) -> types.ModuleType:
    """Read `path` as text, compile it, and execute it in a fresh namespace.

    `allow_package_imports` puts the repository on `sys.path` while the module
    body runs, which a file with `from governor.x import y` needs. It is off by
    default so that a module claiming to be self-contained is actually tested
    as self-contained.
    """
    p = Path(path)
    if not p.exists():
        raise LoaderError(f"no such source file: {p}")
    if not p.is_file():
        raise LoaderError(f"not a file: {p}")

    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise LoaderError(f"cannot read {p} as text: {type(e).__name__}: {e}") from e

    try:
        code = compile(source, str(p), "exec")
    except SyntaxError as e:
        raise LoaderError(f"{p} does not compile: line {e.lineno}: {e.msg}") from e

    module = types.ModuleType(module_name)
    module.__file__ = str(p)
    module.__package__ = module_name.rpartition(".")[0]
    module.__dict__["__source_sha256__"] = hashlib.sha256(source.encode()).hexdigest()

    root = repo_root(p)
    # The module must be visible in `sys.modules` while its body executes.
    # `dataclasses` resolves `sys.modules[cls.__module__].__dict__` to look up
    # string annotations, and `typing` does something similar; with the module
    # absent that lookup returns None and every decorated class raises
    # `AttributeError: 'NoneType' object has no attribute '__dict__'`. The
    # previous entry is restored afterwards so a text load never leaves the
    # normal import path pointing at this namespace.
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        if allow_package_imports:
            with _repo_on_path(root):
                exec(code, module.__dict__)          # noqa: S102 - the whole point
        else:
            exec(code, module.__dict__)              # noqa: S102
    except Exception as e:                            # noqa: BLE001
        raise LoaderError(
            f"executing {p} raised {type(e).__name__}: {e}\n"
            f"  if this module imports from the package, pass "
            f"allow_package_imports=True") from e
    finally:
        if previous is not None:
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
    return module


def load_package_module(dotted: str, root: Path | None = None) -> types.ModuleType:
    """Text-load a module by its dotted package name, resolving the file itself."""
    r = root or repo_root()
    rel = Path(dotted.replace(".", "/") + ".py")
    path = r / rel
    if not path.exists():
        pkg_init = r / dotted.replace(".", "/") / "__init__.py"
        if pkg_init.exists():
            path = pkg_init
        else:
            raise LoaderError(f"cannot resolve {dotted!r} to a file under {r}")
    return load_source_module(path, dotted, allow_package_imports=True)


def public_surface(module: types.ModuleType) -> dict[str, str]:
    """Names and signatures a caller could depend on, for comparing two load paths."""
    out: dict[str, str] = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if inspect.isclass(obj):
            out[name] = f"class:{obj.__name__}"
        elif inspect.isfunction(obj):
            try:
                out[name] = f"def:{inspect.signature(obj)}"
            except (ValueError, TypeError):
                out[name] = "def:<signature unavailable>"
    return out


def verify_import_parity(dotted: str, root: Path | None = None) -> dict:
    """Load a module BOTH ways and require that they agree.

    A divergence means the notebook's normal-import path is reading something
    other than the repository source -- a stale cache, a shadowed name, an
    editable install pointing elsewhere. That is exactly the failure this whole
    file exists to make visible, so it is reported rather than smoothed over.
    """
    r = root or repo_root()
    text_mod = load_package_module(dotted, r)
    with _repo_on_path(r):
        for cached in [m for m in sys.modules if m == dotted or m.startswith(dotted + ".")]:
            sys.modules.pop(cached, None)
        try:
            norm_mod = importlib.import_module(dotted)
        except Exception as e:                        # noqa: BLE001
            return {"module": dotted, "ok": False,
                    "problems": [f"normal import failed: {type(e).__name__}: {e}"]}

    t, n = public_surface(text_mod), public_surface(norm_mod)
    problems: list[str] = []
    only_text, only_norm = sorted(set(t) - set(n)), sorted(set(n) - set(t))
    if only_text:
        problems.append(f"present only via text loader: {only_text[:6]}")
    if only_norm:
        problems.append(f"present only via normal import: {only_norm[:6]}")
    for k in sorted(set(t) & set(n)):
        if t[k] != n[k]:
            problems.append(f"{k}: text {t[k]} vs import {n[k]}")

    text_file = Path(getattr(text_mod, "__file__", ""))
    norm_file = Path(getattr(norm_mod, "__file__", "") or "")
    if norm_file and text_file.resolve() != norm_file.resolve():
        problems.append(f"different files: {text_file} vs {norm_file}")

    return {"module": dotted, "ok": not problems, "problems": problems,
            "names": len(t), "source_sha256": getattr(text_mod, "__source_sha256__", "")}


def main() -> int:
    """`python scripts/colab_text_loader.py` -- self-test on the core modules."""
    targets = [
        "governor.harness.ledger",
        "governor.harness.traps",
        "governor.harness.drivers",
        "governor.gate.m2_interface",
        "governor.execution.executor",
        "governor.phase4.statemgr",
        "governor.execfeedback.sandbox",
        "governor.execfeedback.preflight",
        "governor.execfeedback.publictests",
    ]
    root = repo_root()
    print(f"  repository: {root}")
    bad = 0
    for dotted in targets:
        try:
            r = verify_import_parity(dotted, root)
        except LoaderError as e:
            print(f"  FAIL  {dotted:<40} {e}")
            bad += 1
            continue
        mark = " ok " if r["ok"] else "FAIL"
        print(f"  {mark}  {dotted:<40} {r['names']:>3} names  "
              f"sha {r['source_sha256'][:8]}")
        for p in r["problems"]:
            print(f"        - {p}")
        bad += 0 if r["ok"] else 1
    print(f"\n  {len(targets)-bad}/{len(targets)} modules load identically both ways")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
