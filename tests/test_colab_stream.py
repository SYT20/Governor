"""Streaming must show progress live and prove liveness during silence.

Two failure modes have already cost hours here. `subprocess.run` with inherited
stdout does not reach a notebook cell at all -- IPython captures `sys.stdout` at
the Python level, not the file-descriptor level. `capture_output=True` reaches it
but only after the process exits, so a hung run and a slow one look identical.
A pilot deadlocked for two hours and neither could be distinguished from work.
"""
from __future__ import annotations

import pathlib
import sys
import time

from scripts.colab_stream import run_streamed


def _script(tmp_path: pathlib.Path, body: str) -> str:
    p = tmp_path / "s.py"
    p.write_text(body)
    return str(p)


def test_output_arrives_before_the_process_exits(tmp_path, capsys):
    """The point of streaming: lines appear while it runs, not at the end."""
    s = _script(tmp_path, "import time\nprint('first', flush=True)\n"
                          "time.sleep(1)\nprint('second', flush=True)\n")
    rc = run_streamed([sys.executable, "-u", s], heartbeat_s=10)
    out = capsys.readouterr().out
    assert rc == 0
    assert "first" in out and "second" in out
    assert out.index("first") < out.index("second")


def test_heartbeat_reports_liveness_during_silence(tmp_path, capsys):
    """Silence must be distinguishable from death."""
    s = _script(tmp_path, "import time\nprint('go', flush=True)\ntime.sleep(4)\n")
    run_streamed([sys.executable, "-u", s], heartbeat_s=1.0)
    out = capsys.readouterr().out
    assert "no output for" in out
    assert "process alive" in out


def test_nonzero_exit_is_returned_not_raised(tmp_path, capsys):
    s = _script(tmp_path, "import sys\nprint('failing', flush=True)\nsys.exit(3)\n")
    rc = run_streamed([sys.executable, "-u", s], heartbeat_s=10)
    assert rc == 3
    assert "exit 3" in capsys.readouterr().out


def test_stderr_is_interleaved_not_lost(tmp_path, capsys):
    """A traceback on stderr is usually the only explanation there is."""
    s = _script(tmp_path, "import sys\nsys.stderr.write('boom\\n')\nsys.exit(1)\n")
    run_streamed([sys.executable, "-u", s], heartbeat_s=10)
    assert "boom" in capsys.readouterr().out


def test_timeout_kills_a_hang_and_reports_it(tmp_path, capsys):
    s = _script(tmp_path, "import time\nprint('hanging', flush=True)\n"
                          "time.sleep(300)\n")
    t0 = time.time()
    rc = run_streamed([sys.executable, "-u", s], heartbeat_s=1.0, timeout_s=3.0)
    assert rc == 124
    assert time.time() - t0 < 30
    assert "killed after" in capsys.readouterr().out


def test_summary_line_reports_exit_and_line_count(tmp_path, capsys):
    s = _script(tmp_path, "for i in range(5): print(i, flush=True)\n")
    run_streamed([sys.executable, "-u", s], heartbeat_s=10)
    out = capsys.readouterr().out
    assert "exit 0" in out and "lines" in out
