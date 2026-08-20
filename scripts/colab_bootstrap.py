#!/usr/bin/env python3
"""One-cell bootstrap: clone, pin, install, diagnose, verify. Assumes nothing.

Written to be pasted into a brand-new Colab notebook and run first. It assumes no
working directory, no prior imports, no pip state, no Drive, no PYTHONPATH, and
no cell having run before it.

It ends by printing exactly one of:

    COLAB_BOOTSTRAP = PASS
    COLAB_BOOTSTRAP = FAIL   <reason>

A FAIL is a stop, not a warning. Everything downstream assumes the environment
described in `results/colab_environment.json`, and a partially-installed runtime
produces results that look fine and are not comparable to anything.

Run directly:
    python scripts/colab_bootstrap.py --repo <url> --ref <commit-or-tag>
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "https://github.com/SYT20/Governor.git"
CHECKOUT = Path("/content/Governor") if Path("/content").is_dir() else Path.cwd() / "Governor"


def sh(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> {r.returncode}\n{r.stderr[:500]}")
    return r.stdout.strip()


def hardware() -> dict:
    """Everything the run's comparability depends on. GPU absence is reported, not assumed."""
    info: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG", "not-colab"),
        "in_colab": "google.colab" in sys.modules or Path("/content").is_dir(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
        info["ram_available_gb"] = round(vm.available / 1e9, 2)
    except Exception:                                     # noqa: BLE001
        info["ram_total_gb"] = None
    try:
        du = shutil.disk_usage("/content" if Path("/content").is_dir() else ".")
        info["disk_free_gb"] = round(du.free / 1e9, 2)
    except Exception:                                     # noqa: BLE001
        info["disk_free_gb"] = None

    info["gpu"] = {"present": False}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(i)
            info["gpu"] = {
                "present": True, "name": props.name,
                "memory_gb": round(props.total_memory / 1e9, 2),
                "capability": f"{props.major}.{props.minor}",
                "cuda": torch.version.cuda, "count": torch.cuda.device_count(),
            }
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpu"] = {"present": True, "name": "Apple MPS", "memory_gb": None,
                           "cuda": None, "count": 1}
    except Exception as e:                                # noqa: BLE001
        info["torch"] = f"unavailable: {type(e).__name__}"
    return info


def package_versions() -> dict:
    out = {}
    for mod in ("numpy", "scipy", "sklearn", "pandas", "torch", "transformers",
                "huggingface_hub", "datasets", "psutil", "pytest", "accelerate"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:                            # noqa: BLE001
            out[mod] = f"MISSING ({type(e).__name__})"
    return out


def acquire(repo: str, ref: str, dest: Path) -> dict:
    """Clone at a frozen ref. A moving target is not a reproducible experiment."""
    if dest.exists() and (dest / ".git").is_dir():
        sh(["git", "fetch", "--all", "--tags", "--quiet"], cwd=dest, check=False)
    else:
        if dest.exists():
            shutil.rmtree(dest)
        sh(["git", "clone", "--quiet", repo, str(dest)])
    if ref:
        sh(["git", "checkout", "--quiet", ref], cwd=dest)
    head = sh(["git", "rev-parse", "HEAD"], cwd=dest)
    try:
        described = sh(["git", "describe", "--tags", "--always"], cwd=dest)
    except RuntimeError:
        described = head[:12]
    # Reproducibility means the TRACKED SOURCE matches the recorded commit.
    # Untracked files do not: the notebook legitimately writes outputs into the
    # checkout as it runs, so counting them made bootstrap fail on its own
    # exhaust the moment the cell was re-run. Modified tracked files are the
    # real hazard and still fail.
    modified = sh(["git", "status", "--porcelain", "--untracked-files=no"],
                  cwd=dest, check=False)
    untracked = sh(["git", "status", "--porcelain", "--untracked-files=all"],
                   cwd=dest, check=False)
    n_untracked = len([l for l in untracked.splitlines() if l.startswith("??")])
    return {"repo": repo, "requested_ref": ref, "commit": head,
            "describe": described, "dirty": bool(modified.strip()),
            "modified_tracked": [l.strip() for l in modified.splitlines()][:10],
            "untracked_count": n_untracked, "path": str(dest)}


def install(root: Path) -> dict:
    req = root / "configs" / "colab-requirements.txt"
    if not req.exists():
        return {"ok": False, "error": f"missing {req}"}
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                       capture_output=True, text=True)
    return {"ok": r.returncode == 0, "returncode": r.returncode,
            "stderr_tail": r.stderr[-800:] if r.returncode else ""}


def import_smoke(root: Path) -> dict:
    """Prove the repository imports from a subprocess, not from this session."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import governor.harness.ledger as L\n"
        "import governor.harness.traps as T\n"
        "import governor.gate.m2_interface as M\n"
        "import governor.execfeedback.sandbox as S\n"
        "print('OK', len(dir(L)), len(dir(T)), len(dir(M)), len(dir(S)))\n"
    ) % str(root)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return {"ok": r.returncode == 0, "stdout": r.stdout.strip(),
            "stderr_tail": r.stderr[-600:] if r.returncode else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--ref", default="", help="commit or tag; empty means default branch")
    ap.add_argument("--dest", default=str(CHECKOUT))
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    problems: list[str] = []
    print("=" * 66)
    print("GOVERNOR — COLAB BOOTSTRAP")
    print("=" * 66)

    hw = hardware()
    print("\n[01] environment")
    for k in ("python", "platform", "machine", "cpu_count", "ram_total_gb",
              "disk_free_gb", "colab_release_tag", "in_colab"):
        print(f"       {k:<20} {hw.get(k)}")
    g = hw["gpu"]
    if g["present"]:
        mem = f" ({g['memory_gb']} GB)" if g.get("memory_gb") else ""
        gpu_line = f"{g['name']}{mem}  cuda={g.get('cuda')}"
    else:
        gpu_line = "NONE"
    print(f"       {'gpu':<20} {gpu_line}")
    if not g["present"]:
        print("       note: no GPU. Not fatal here; the notebook decides whether it needs one.")

    print("\n[02] repository")
    try:
        repo = acquire(args.repo, args.ref, Path(args.dest))
        print(f"       commit   {repo['commit']}")
        print(f"       describe {repo['describe']}")
        print(f"       path     {repo['path']}")
        if repo["untracked_count"]:
            print(f"       {'untracked':<20} {repo['untracked_count']} file(s) — run "
                  f"outputs, not a reproducibility problem")
        if repo["dirty"]:
            problems.append(
                f"TRACKED files modified, so the commit does not describe the code: "
                f"{repo['modified_tracked'][:5]}")
    except Exception as e:                                # noqa: BLE001
        print(f"       FAILED: {type(e).__name__}: {e}")
        print("\nCOLAB_BOOTSTRAP = FAIL   repository acquisition failed")
        return 1

    root = Path(repo["path"])
    print("\n[03] dependencies")
    if args.skip_install:
        print("       skipped by request")
        inst = {"ok": True, "skipped": True}
    else:
        inst = install(root)
        print(f"       pip install -r configs/colab-requirements.txt -> "
              f"{'ok' if inst['ok'] else 'FAILED'}")
        if not inst["ok"]:
            print(inst.get("stderr_tail", ""))
            problems.append("dependency installation failed")

    pkgs = package_versions()
    print("\n[04] versions")
    for k, v in pkgs.items():
        print(f"       {k:<18} {v}")
        if str(v).startswith("MISSING") and k in ("numpy", "sklearn", "torch",
                                                  "transformers", "psutil"):
            problems.append(f"required package missing: {k}")

    print("\n[05] import smoke (subprocess, not this session)")
    smoke = import_smoke(root)
    print(f"       {smoke['stdout'] or smoke.get('stderr_tail','')[:200]}")
    if not smoke["ok"]:
        problems.append("repository does not import in a fresh subprocess")

    env = {"hardware": hw, "packages": pkgs, "repository": repo,
           "install": inst, "import_smoke": smoke, "problems": problems}
    out = root / "results" / "colab_environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=1))
    print(f"\n[06] wrote {out}")

    print("\n" + "=" * 66)
    if problems:
        for p in problems:
            print(f"  - {p}")
        print(f"COLAB_BOOTSTRAP = FAIL   {len(problems)} problem(s)")
        return 1
    print("COLAB_BOOTSTRAP = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
