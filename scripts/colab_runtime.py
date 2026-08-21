#!/usr/bin/env python3
"""Detect which notebook runtime this is, and what it can actually do.

The notebook has to run identically under hosted Colab and under the VS Code
Colab/Jupyter extension. Those differ in ways that break naive code:

  google.colab      importable in hosted Colab; frequently absent or non-functional
                    under VS Code even when the compute is a Colab VM
  Drive OAuth       needs the kernel's interactive channel, which VS Code does not
                    necessarily provide
  /content          present on a Colab VM regardless of which client is attached

Every one of those was previously inferred from a single variable, which is how a
notebook ends up asserting "not running in Colab" on a machine that plainly is.
Detection here uses several independent signals and reports what it saw, so a
wrong answer is auditable rather than mysterious.

THIS FILE DETECTS; IT DOES NOT DECIDE POLICY. It reports what Drive can do here
and never mounts, because mounting needs the notebook kernel.

The rule this file used to state -- "Drive is an ARCHIVE, never a dependency,
and a Drive that cannot mount is recorded as UNAVAILABLE, not an experiment
failure" -- was WRONG, and it cost a run. E0029 generated 4750 samples in hosted
Colab, the mount raised ValueError, the status read SKIPPED, the notebook
printed ARCHIVE_STATUS = PASS because Drive was "not required", and the VM was
destroyed with the only copy on it.

A durable copy is a PRECONDITION of expensive, unrepeatable work, not an
optional epilogue. scripts/durable_sink.py enforces that: verify a sink by
round-trip before the model loads, mirror at every checkpoint, and refuse to
start without one. What stays true is narrower and still worth saying: this
module must not turn a detection result into a raised exception.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass, asdict, field

HOSTED_COLAB = "HOSTED_COLAB"
VSCODE_COLAB = "VSCODE_COLAB"
LOCAL_JUPYTER = "LOCAL_JUPYTER"
UNKNOWN = "UNKNOWN"


@dataclass
class Runtime:
    kind: str
    python: str
    executable: str
    cwd: str
    repo: str
    commit: str
    has_google_colab: bool
    has_content_dir: bool
    has_drive_dir: bool
    in_ipython: bool
    ipython_class: str
    colab_release_tag: str
    vscode_markers: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        return "\n".join([
            f"  runtime        {self.kind}",
            f"  python         {self.python}  ({self.executable})",
            f"  cwd            {self.cwd}",
            f"  repo           {self.repo}",
            f"  commit         {self.commit}",
            f"  google.colab   {self.has_google_colab}",
            f"  /content       {self.has_content_dir}",
            f"  /content/drive {self.has_drive_dir}",
            f"  ipython        {self.ipython_class or 'not in IPython'}",
            f"  colab tag      {self.colab_release_tag}",
            f"  vscode markers {self.vscode_markers or 'none'}",
        ])


def _git_head(path: pathlib.Path) -> str:
    """Never raise from a diagnostic: subprocess.run(cwd=...) raises when the
    directory is absent, which is exactly what this is meant to report."""
    if not path.is_dir():
        return "(directory does not exist)"
    try:
        r = subprocess.run(["git", "log", "--oneline", "-1"], cwd=str(path),
                           capture_output=True, text=True)
        return r.stdout.strip() or "(not a git repo)"
    except Exception as e:                                # noqa: BLE001
        return f"(git unavailable: {type(e).__name__})"


def find_repo(start: pathlib.Path | None = None) -> pathlib.Path:
    """Locate the checkout by marker, not by assuming a working directory."""
    cwd = start or pathlib.Path.cwd()
    for cand in (cwd, cwd / "Governor", *cwd.parents):
        if (cand / "governor").is_dir() and (cand / "Makefile").exists():
            return cand
    return cwd / "Governor"


def detect_notebook_runtime() -> Runtime:
    """Classify the runtime from several independent signals, never one."""
    signals: dict = {}

    try:
        import google.colab                               # noqa: F401
        has_colab = True
    except Exception:                                     # noqa: BLE001
        has_colab = False
    signals["google_colab_import"] = has_colab

    has_content = pathlib.Path("/content").is_dir()
    has_drive = pathlib.Path("/content/drive/MyDrive").is_dir()
    signals["content_dir"] = has_content
    signals["drive_dir"] = has_drive

    ipy_class, in_ipy = "", False
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            in_ipy = True
            ipy_class = type(ip).__name__
    except Exception:                                     # noqa: BLE001
        pass
    signals["ipython_class"] = ipy_class

    # VS Code sets these in the KERNEL's environment; hosted Colab does not.
    # TERM_PROGRAM is deliberately excluded: it describes whichever terminal
    # launched the process, not what is driving the kernel, and it is inherited
    # by anything started from a VS Code terminal -- including a plain shell on a
    # developer machine. Using it misclassified a hosted-Colab simulation.
    vscode_markers = [k for k in ("VSCODE_PID", "VSCODE_CWD", "VSCODE_IPC_HOOK",
                                  "VSCODE_IPC_HOOK_CLI")
                      if os.environ.get(k)]
    signals["vscode_markers"] = vscode_markers

    tag = os.environ.get("COLAB_RELEASE_TAG", "")
    signals["colab_release_tag"] = tag or "(unset)"

    # Weigh the signals. A Colab VM driven from VS Code shows /content and often
    # COLAB_RELEASE_TAG, but google.colab may be absent or inert -- so neither
    # alone decides.
    colab_vm = has_content or bool(tag)
    if colab_vm and has_colab and not vscode_markers:
        kind = HOSTED_COLAB
    elif colab_vm and (vscode_markers or not has_colab):
        kind = VSCODE_COLAB
    elif in_ipy and not colab_vm:
        kind = LOCAL_JUPYTER
    elif not in_ipy:
        kind = LOCAL_JUPYTER if not colab_vm else VSCODE_COLAB
    else:
        kind = UNKNOWN

    repo = find_repo()
    return Runtime(
        kind=kind, python=sys.version.split()[0], executable=sys.executable,
        cwd=str(pathlib.Path.cwd()), repo=str(repo), commit=_git_head(repo),
        has_google_colab=has_colab, has_content_dir=has_content,
        has_drive_dir=has_drive, in_ipython=in_ipy, ipython_class=ipy_class,
        colab_release_tag=tag or "(unset)", vscode_markers=vscode_markers,
        signals=signals)


@dataclass
class DriveStatus:
    available: bool
    mounted: bool
    path: str
    interactive_mount_supported: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def detect_drive(runtime: Runtime | None = None) -> DriveStatus:
    """Report what Drive can do here. Never mounts; mounting needs the kernel."""
    rt = runtime or detect_notebook_runtime()
    mnt = pathlib.Path("/content/drive/MyDrive")

    if mnt.is_dir():
        writable, why = _probe_writable(mnt)
        return DriveStatus(available=writable, mounted=True, path=str(mnt),
                           interactive_mount_supported=rt.has_google_colab,
                           reason=why)

    if rt.kind == HOSTED_COLAB and rt.has_google_colab:
        return DriveStatus(False, False, str(mnt), True,
                           "not mounted; hosted Colab can mount interactively")
    if rt.kind == VSCODE_COLAB:
        return DriveStatus(
            False, False, str(mnt), rt.has_google_colab,
            "not mounted. Under the VS Code extension the OAuth prompt often has "
            "nowhere to appear, so mounting may be impossible here. The archive "
            "runs locally regardless.")
    return DriveStatus(False, False, str(mnt), False,
                       f"no Drive on this runtime ({rt.kind})")


def _probe_writable(mnt: pathlib.Path) -> tuple[bool, str]:
    """A stale or half-detached mount still presents the directory, so write."""
    probe = mnt / ".governor_write_probe"
    try:
        probe.write_text("ok")
        got = probe.read_text()
        probe.unlink()
    except Exception as e:                                # noqa: BLE001
        return False, f"mounted but not writable: {type(e).__name__}"
    return (got == "ok"), ("mounted and writable" if got == "ok"
                           else "probe read back wrong -- mount unhealthy")


@dataclass
class GpuStatus:
    available: bool
    name: str
    memory_gb: float | None
    cuda: str
    backend: str
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def detect_gpu() -> GpuStatus:
    """Actual hardware, independent of which client is attached.

    REQUIRE_GPU must fail on the ABSENCE OF A GPU, never on a guess about the
    runtime -- a Colab VM driven from VS Code has exactly the same card.
    """
    try:
        import torch
    except ImportError:
        return GpuStatus(False, "", None, "", "none", "torch is not installed")
    try:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return GpuStatus(True, p.name, round(p.total_memory / 1e9, 2),
                             torch.version.cuda or "", "cuda", "ok")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return GpuStatus(True, "Apple MPS", None, "", "mps", "ok")
    except Exception as e:                                # noqa: BLE001
        return GpuStatus(False, "", None, "", "none",
                         f"probe failed: {type(e).__name__}: {e}")
    return GpuStatus(False, "", None, "", "none", "no CUDA or MPS device visible")


def main() -> int:
    rt = detect_notebook_runtime()
    print(rt.render())
    d = detect_drive(rt)
    g = detect_gpu()
    print(f"\n  DRIVE   available={d.available} mounted={d.mounted} "
          f"interactive={d.interactive_mount_supported}")
    print(f"          {d.reason}")
    print(f"\n  GPU     available={g.available} {g.name or '-'} "
          f"{g.memory_gb or ''} backend={g.backend}")
    print(f"          {g.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
