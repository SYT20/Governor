"""An untrusted-code execution boundary, not a subprocess wrapper.

E0026 runs thousands of model-generated programs. Treating that as
`subprocess.run(..., timeout=n)` is inadequate in three specific ways, each of
which was hit while building this:

  1. `timeout=` kills the direct child. A program that forks leaves orphans
     holding CPU, and the next measurement inherits a loaded machine. Every run
     therefore starts a new session and the whole PROCESS GROUP is signalled.
  2. Without a memory limit, one allocation can drive the host into swap and
     stall every other measurement -- so latency stops meaning anything. The
     limit is RLIMIT_DATA everywhere and RLIMIT_AS only on macOS: on Linux the
     address-space cap counts virtual reservations CPython never makes resident,
     so a useful value there can stop `print("hi")` from running.
  3. Unbounded stdout from a looping print fills the pipe buffer, the child
     blocks on write, and the timeout fires. A capped writer records this as
     OUTPUT_OVERFLOW instead of mislabelling it a timeout.

Nothing here interprets correctness. It reports what happened when the program
ran; whether that constitutes a passing solution is the evaluator's business and
is deliberately kept on the other side of this boundary.
"""
from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_TIMEOUT_S = 6.0
DEFAULT_MEM_MB = 1024
DEFAULT_OUTPUT_CAP = 64_000
MAX_PROCESSES = 64


class Status:
    OK = "OK"
    COMPILE_ERROR = "COMPILE_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"
    MEMORY = "MEMORY"
    OUTPUT_OVERFLOW = "OUTPUT_OVERFLOW"
    LAUNCH_FAILED = "LAUNCH_FAILED"


@dataclass(frozen=True)
class RunResult:
    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    latency_s: float
    truncated: bool

    @property
    def ok(self) -> bool:
        return self.status == Status.OK

    def as_dict(self) -> dict:
        return asdict(self)


def _limits(mem_mb: int, cpu_s: int, fsize_bytes: int = 64 * 1024 * 1024):
    """Applied in the child, between fork and exec."""
    def apply():
        os.setsid()                                     # own process group
        soft = mem_mb * 1024 * 1024
        # RLIMIT_AS caps VIRTUAL address space, and on Linux glibc and the CPython
        # runtime reserve large virtual regions that are never resident -- a cap
        # tight enough to be useful there can make even `print("hi")` fail, while
        # macOS frequently does not enforce it at all. So limit the HEAP
        # (RLIMIT_DATA) everywhere, and only add the address-space cap on
        # platforms where it behaves. Which limits actually took effect is
        # recorded rather than assumed.
        applied = []
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (soft, soft))
            applied.append("DATA")
        except (ValueError, OSError):
            pass
        if sys.platform == "darwin":
            try:
                resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
                applied.append("AS")
            except (ValueError, OSError):
                pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass
        try:
            # Bounds the output files from the writing side, so a flooding
            # program is stopped by the kernel rather than by disk exhaustion.
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
        except (ValueError, OSError):
            pass
    return apply


def _kill_group(proc: subprocess.Popen) -> None:
    """Signal the whole group; a forking program outlives a kill on the child."""
    for sig in (signal.SIGKILL,):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def check_syntax(code: str) -> tuple[bool, str]:
    """Parse without executing. A file that cannot compile never reaches a shell."""
    try:
        compile(code, "<candidate>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"{type(e).__name__}: {e.msg} (line {e.lineno})"
    except (ValueError, MemoryError, RecursionError) as e:
        return False, f"{type(e).__name__}: {e}"


def run(code: str, stdin: str = "", *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        mem_mb: int = DEFAULT_MEM_MB,
        output_cap: int = DEFAULT_OUTPUT_CAP) -> RunResult:
    """Execute `code` with `stdin`, and report what happened.

    Never raises for a misbehaving program -- a crash, a hang and an allocation
    storm are all outcomes to be recorded, not exceptions to be handled by the
    caller.
    """
    syntax_ok, syntax_msg = check_syntax(code)
    if not syntax_ok:
        return RunResult(Status.COMPILE_ERROR, None, "", syntax_msg, 0.0, False)

    with tempfile.TemporaryDirectory(prefix="e0026_") as td:
        src = Path(td) / "candidate.py"
        src.write_text(code)

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": td,
            "TMPDIR": td,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",           # determinism across repeats
        }

        # Output goes to FILES, not pipes. `communicate()` accumulates the child's
        # output in this process, so a program that floods stdout quickly can
        # exhaust the PARENT's memory long before `output_cap` is ever applied --
        # which at several thousand executions is a real hazard, not a corner
        # case. Files are disk-backed and bounded below by RLIMIT_FSIZE, and a
        # file never blocks the writer the way a full pipe does.
        out_path, err_path = Path(td) / "stdout", Path(td) / "stderr"
        started = time.perf_counter()
        try:
            with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
                proc = subprocess.Popen(
                    [sys.executable, "-I", "-S", str(src)],
                    stdin=subprocess.PIPE, stdout=fo, stderr=fe,
                    cwd=td, env=env,
                    preexec_fn=_limits(mem_mb, int(timeout_s) + 1, output_cap * 64),
                )
                try:
                    proc.stdin.write(stdin.encode())
                except (BrokenPipeError, OSError):
                    pass                     # program never read stdin; not an error
                finally:
                    try:
                        proc.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
        except (OSError, ValueError) as e:
            return RunResult(Status.LAUNCH_FAILED, None, "", str(e)[:400], 0.0, False)

        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None:
                _kill_group(proc)

        latency = time.perf_counter() - started

        def _read(p: Path) -> tuple[str, bool]:
            try:
                size = p.stat().st_size
                with open(p, "rb") as f:
                    data = f.read(output_cap)
                return data.decode("utf-8", "replace"), size > output_cap
            except OSError:
                return "", False

        out, out_trunc = _read(out_path)
        err, err_trunc = _read(err_path)
        truncated = out_trunc or err_trunc

        if timed_out:
            return RunResult(Status.TIMEOUT, None, out, err, latency, truncated)

        code_ = proc.returncode
        if code_ == 0:
            status = Status.OUTPUT_OVERFLOW if truncated else Status.OK
        elif code_ in (-signal.SIGKILL, -signal.SIGXCPU, 137):
            status = Status.TIMEOUT
        elif "MemoryError" in err:
            status = Status.MEMORY
        else:
            status = Status.RUNTIME_ERROR

        return RunResult(status, code_, out, err, latency, truncated)
