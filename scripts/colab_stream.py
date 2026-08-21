#!/usr/bin/env python3
"""Run a subprocess so that a Jupyter/Colab cell can actually see it working.

THE PROBLEM THIS SOLVES

`subprocess.run(cmd)` with inherited stdout does not stream inside a notebook.
IPython replaces `sys.stdout` at the PYTHON level, not at the file-descriptor
level, so a child process writing to fd 1 bypasses the capture entirely: its
output goes to the kernel log, which the user never sees. Adding `-u` does not
help -- the child is unbuffered and the bytes still go somewhere invisible.

`capture_output=True` has the mirror defect: it holds everything until the
process exits, so a run taking minutes shows nothing and a HUNG run is
indistinguishable from a working one.

Both failure modes have already happened here. A pilot deadlocked for two hours
having printed one line, and there was no way to tell it apart from slow
progress.

THE FIX

Read the child's pipe line by line and re-emit each line through Python's own
`print`, which IPython does capture. Add a heartbeat so silence is distinguished
from death: if nothing arrives for a while, say so, with elapsed time, rather
than leaving a blank cell that means nothing.

    from scripts.colab_stream import run_streamed
    rc = run_streamed([sys.executable, "script.py", "--flag"])
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from queue import Empty, Queue


def run_streamed(cmd: list[str], *, heartbeat_s: float = 30.0,
                 timeout_s: float | None = None, prefix: str = "") -> int:
    """Run `cmd`, echoing its output live into the notebook. Returns the exit code.

    `heartbeat_s` controls how long silence may last before a keep-alive line is
    printed. That line is the whole point: it is the difference between "this is
    working and slow" and "this has hung", which is otherwise unknowable from
    inside a notebook cell.
    """
    started = time.time()
    print(f"{prefix}$ {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True)

    q: Queue = Queue()

    def pump() -> None:
        try:
            for line in proc.stdout:                     # type: ignore[union-attr]
                q.put(line)
        finally:
            q.put(None)

    threading.Thread(target=pump, daemon=True).start()

    last_output = time.time()
    lines = 0
    while True:
        if timeout_s and time.time() - started > timeout_s:
            proc.kill()
            print(f"{prefix}[killed after {timeout_s:.0f}s]", flush=True)
            return 124
        try:
            item = q.get(timeout=min(heartbeat_s, 5.0))
        except Empty:
            quiet = time.time() - last_output
            if quiet >= heartbeat_s:
                alive = proc.poll() is None
                print(f"{prefix}[{time.time()-started:>6.0f}s elapsed, no output for "
                      f"{quiet:.0f}s, process {'alive' if alive else 'exited'}]",
                      flush=True)
                last_output = time.time()
            continue
        if item is None:
            break
        print(f"{prefix}{item}", end="", flush=True)
        last_output = time.time()
        lines += 1

    proc.wait()
    print(f"{prefix}[exit {proc.returncode} after {time.time()-started:.0f}s, "
          f"{lines} lines]", flush=True)
    return proc.returncode


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    return run_streamed(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
