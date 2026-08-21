"""Runtime detection must classify from several signals, never one.

The notebook runs under hosted Colab and under the VS Code Colab extension, and
those differ in ways that break naive checks: google.colab may be absent under
VS Code even when the compute IS a Colab VM, and Drive OAuth needs an interactive
channel VS Code does not necessarily provide. Inferring from a single variable is
how a notebook comes to assert "not running in Colab" on a machine that plainly
is -- which happened here.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

from scripts import colab_runtime as cr


def _env(monkeypatch, *, colab_import: bool, content: bool, drive: bool,
         vscode: bool, tag: str = ""):
    """Simulate one runtime by controlling every signal detection reads."""
    # google.colab is not installed on a developer machine, so "present" has to
    # be simulated by injecting a stub rather than by letting the real import
    # through -- which would fail and make the hosted-Colab case untestable here.
    import types
    if colab_import:
        pkg = types.ModuleType("google")
        sub = types.ModuleType("google.colab")
        sub.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
        pkg.colab = sub
        monkeypatch.setitem(sys.modules, "google", pkg)
        monkeypatch.setitem(sys.modules, "google.colab", sub)
    else:
        monkeypatch.setitem(sys.modules, "google.colab", None)   # forces ImportError

    real_is_dir = pathlib.Path.is_dir

    def fake_is_dir(self):
        s = str(self)
        if s == "/content":
            return content
        if s == "/content/drive/MyDrive":
            return drive
        return real_is_dir(self)

    monkeypatch.setattr(pathlib.Path, "is_dir", fake_is_dir)

    for k in ("VSCODE_PID", "VSCODE_CWD", "VSCODE_IPC_HOOK", "VSCODE_IPC_HOOK_CLI"):
        monkeypatch.delenv(k, raising=False)
    if vscode:
        monkeypatch.setenv("VSCODE_PID", "1234")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    if tag:
        monkeypatch.setenv("COLAB_RELEASE_TAG", tag)


def test_hosted_colab(monkeypatch):
    _env(monkeypatch, colab_import=True, content=True, drive=False, vscode=False,
         tag="release-colab-external-images_20260819")
    assert cr.detect_notebook_runtime().kind == cr.HOSTED_COLAB


def test_vscode_driving_a_colab_vm_without_google_colab(monkeypatch):
    """The case that broke the notebook: a Colab VM where google.colab is absent."""
    _env(monkeypatch, colab_import=False, content=True, drive=False, vscode=True)
    rt = cr.detect_notebook_runtime()
    assert rt.kind == cr.VSCODE_COLAB
    assert rt.has_content_dir and not rt.has_google_colab


def test_vscode_driving_a_colab_vm_with_google_colab(monkeypatch):
    _env(monkeypatch, colab_import=True, content=True, drive=False, vscode=True)
    assert cr.detect_notebook_runtime().kind == cr.VSCODE_COLAB


def test_local_machine(monkeypatch):
    _env(monkeypatch, colab_import=False, content=False, drive=False, vscode=True)
    assert cr.detect_notebook_runtime().kind == cr.LOCAL_JUPYTER


def test_drive_is_never_required(monkeypatch):
    """Drive unavailable must be a status, not an exception."""
    _env(monkeypatch, colab_import=False, content=True, drive=False, vscode=True)
    d = cr.detect_drive()
    assert d.available is False and d.mounted is False
    assert isinstance(d.reason, str) and d.reason


def test_drive_mounted_is_probed_not_assumed(monkeypatch, tmp_path):
    """A stale mount still presents the directory, so availability needs a write."""
    _env(monkeypatch, colab_import=True, content=True, drive=True, vscode=False)
    monkeypatch.setattr(cr, "_probe_writable",
                        lambda p: (False, "mounted but not writable: OSError"))
    d = cr.detect_drive()
    assert d.mounted is True and d.available is False


def test_gpu_detection_is_independent_of_runtime_kind():
    """REQUIRE_GPU must fail on absent hardware, not on a guess about the client."""
    g = cr.detect_gpu()
    assert isinstance(g.available, bool)
    assert g.reason                      # always explains itself
    if not g.available:
        assert g.backend == "none"


def test_detection_never_raises_without_a_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rt = cr.detect_notebook_runtime()          # must not raise
    assert rt.kind in (cr.HOSTED_COLAB, cr.VSCODE_COLAB,
                       cr.LOCAL_JUPYTER, cr.UNKNOWN)
    assert "does not exist" in rt.commit or "not a git repo" in rt.commit


def test_signals_are_recorded_for_audit():
    rt = cr.detect_notebook_runtime()
    for k in ("google_colab_import", "content_dir", "vscode_markers"):
        assert k in rt.signals
