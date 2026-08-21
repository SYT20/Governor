#!/usr/bin/env python3
"""A place to write that outlives the VM -- verified before the GPU starts.

WHAT WENT WRONG, so it cannot go wrong the same way twice.

E0029 generated 4750 samples over hours of L4 time. The rows were checkpointed
every batch, resumably, to the VM's local disk. The Drive copy was an ARCHIVE
STEP AT THE END, and it was designed so that a Drive failure could never fail
the experiment. The mount raised ValueError under the VS Code extension, the
cell printed DRIVE_ARCHIVE: SKIPPED, the run "succeeded", the VM was recycled,
and every row was lost.

Two separate mistakes, and the smaller one gets more attention than it deserves:

  1. the mount failed  -- fixable, and largely a VS Code limitation;
  2. persistence was optional, and it happened LAST.

The second is the real defect. Archiving at the end loses everything to a
disconnect mid-run just as surely as to a recycled VM, and "never let the
archive fail the experiment" is exactly backwards when the artifact IS the
experiment. Generation is the expensive, unrepeatable part; grading and analysis
are minutes of CPU and can be redone forever.

So this module inverts it:

  VERIFY FIRST    the sink is written, read back and compared BEFORE any model
                  loads. A sink that cannot round-trip is not a sink.
  MIRROR ALWAYS   rows are appended to the durable copy at every checkpoint, not
                  at the end. Incrementally -- only the new bytes -- so a slow
                  filesystem costs a moment per batch, not a recopy of the file.
  REFUSE LOUDLY   no durable sink means the expensive cell does not start. The
                  override exists and is explicit, so risking it is a decision
                  someone makes rather than a default they never saw.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import time
from dataclasses import dataclass, asdict


@dataclass
class SinkStatus:
    ok: bool
    path: str
    kind: str
    reason: str
    round_trip_ms: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class NoDurableSink(SystemExit):
    """Raised instead of starting work whose only copy would die with the VM."""


def verify_writable(directory: pathlib.Path) -> SinkStatus:
    """Round-trip a probe. A mounted-but-broken Drive still presents a directory,
    so existence proves nothing; only writing and reading back does."""
    kind = _classify(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:                                  # noqa: BLE001
        return SinkStatus(False, str(directory), kind,
                          f"cannot create: {type(e).__name__}: {e}")

    probe = directory / f".governor_probe_{os.getpid()}"
    payload = hashlib.sha256(str(time.time()).encode()).hexdigest()
    t0 = time.perf_counter()
    try:
        probe.write_text(payload)
        got = probe.read_text()
        probe.unlink()
    except Exception as e:                                  # noqa: BLE001
        return SinkStatus(False, str(directory), kind,
                          f"probe failed: {type(e).__name__}: {e}")
    ms = int((time.perf_counter() - t0) * 1000)
    if got != payload:
        return SinkStatus(False, str(directory), kind,
                          "probe read back wrong -- the mount is unhealthy", ms)
    return SinkStatus(True, str(directory), kind, "verified round-trip", ms)


def _classify(directory: pathlib.Path) -> str:
    s = str(directory)
    if s.startswith("/content/drive"):
        return "google-drive"
    if s.startswith("/content"):
        return "vm-local (NOT durable)"
    return "filesystem"


def candidates() -> list[pathlib.Path]:
    """Where a durable copy could live, best first."""
    out = []
    env = os.environ.get("GOVERNOR_SINK")
    if env:
        out.append(pathlib.Path(env))
    out.append(pathlib.Path("/content/drive/MyDrive/governor_e0029"))
    return out


def find_durable_sink() -> tuple[SinkStatus | None, list[SinkStatus]]:
    """First candidate that round-trips wins. Every attempt is reported."""
    tried = []
    for c in candidates():
        st = verify_writable(c)
        tried.append(st)
        if st.ok and "NOT durable" not in st.kind:
            return st, tried
    return None, tried


def require_durable_sink(*, allow_ephemeral: bool = False,
                         quiet: bool = False) -> pathlib.Path | None:
    """Return a verified durable directory, or refuse to let the caller proceed.

    `allow_ephemeral` is the explicit override. It is not a default and it says
    plainly what is being accepted, because the alternative -- a flag whose
    meaning is 'silently lose the run if anything goes wrong' -- is how this
    failed the first time.
    """
    sink, tried = find_durable_sink()
    if not quiet:
        print("  durable sink check:")
        for st in tried:
            mark = "OK  " if st.ok else "FAIL"
            print(f"    [{mark}] {st.path}  ({st.kind})")
            print(f"           {st.reason}")
    if sink is not None:
        if not quiet:
            print(f"    -> writing durable copies to {sink.path} "
                  f"({sink.round_trip_ms} ms round-trip)\n")
        return pathlib.Path(sink.path)

    if allow_ephemeral:
        if not quiet:
            print("    -> NO DURABLE SINK. Proceeding because --allow-ephemeral\n"
                  "       was passed. If this VM is recycled or disconnects, the\n"
                  "       generated rows are GONE and the GPU time is spent for\n"
                  "       nothing. That happened on the previous run.\n")
        return None

    raise NoDurableSink(
        "\nREFUSING TO START: there is nowhere durable to write.\n"
        "\n"
        "  This run produces its expensive artifact on a machine that will be\n"
        "  recycled. Without a verified off-VM copy, finishing successfully and\n"
        "  losing everything are the same outcome -- which is what happened to\n"
        "  the previous run: 4750 samples, hours of L4 time, all lost.\n"
        "\n"
        "  Pick one:\n"
        "\n"
        "  1. MOUNT DRIVE (best). In HOSTED Colab -- colab.research.google.com in\n"
        "     a browser, not the VS Code extension -- run:\n"
        "         from google.colab import drive\n"
        "         drive.mount('/content/drive')\n"
        "     then re-run this cell. The OAuth prompt needs a browser; under the\n"
        "     VS Code extension it often has nowhere to appear, which is exactly\n"
        "     how the mount failed last time.\n"
        "\n"
        "  2. NAME YOUR OWN SINK, if you have a persistent volume mounted:\n"
        "         import os; os.environ['GOVERNOR_SINK'] = '/some/persistent/dir'\n"
        "\n"
        "  3. ACCEPT THE RISK, explicitly, having read the above:\n"
        "         --allow-ephemeral        (script)\n"
        "         ALLOW_EPHEMERAL = True   (notebook)\n"
        "     Then download results/ yourself the moment the run finishes.\n")


class MirroredFile:
    """Appends to a local file and mirrors new bytes to the durable copy.

    Incremental on purpose. Recopying a growing JSONL at every checkpoint turns
    a per-batch cost into a quadratic one, and on Drive that is slow enough that
    someone eventually 'optimises' it by moving the copy to the end of the run --
    which is the original bug.
    """

    def __init__(self, local: pathlib.Path, sink: pathlib.Path | None):
        self.local = pathlib.Path(local)
        self.local.parent.mkdir(parents=True, exist_ok=True)
        self.remote = (pathlib.Path(sink) / self.local.name) if sink else None
        self._mirrored = 0
        if self.remote is not None and self.remote.exists():
            # Resuming: adopt whichever copy is longer. The durable one can be
            # ahead if the VM died after a mirror but before a local flush.
            if not self.local.exists() or \
                    self.remote.stat().st_size > self.local.stat().st_size:
                shutil.copy2(self.remote, self.local)
            self._mirrored = self.local.stat().st_size

    def sync(self) -> int:
        """Copy bytes written since the last sync. Returns bytes transferred."""
        if self.remote is None or not self.local.exists():
            return 0
        size = self.local.stat().st_size
        if size <= self._mirrored:
            return 0
        with open(self.local, "rb") as src:
            src.seek(self._mirrored)
            chunk = src.read()
        with open(self.remote, "ab") as dst:
            dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        self._mirrored = size
        return len(chunk)

    def verify(self) -> tuple[bool, str]:
        """Compare the two copies byte for byte. A mirror nobody checked is a
        belief, not a backup."""
        if self.remote is None:
            return False, "no durable copy (ephemeral run)"
        if not self.remote.exists():
            return False, "durable copy missing"
        a = hashlib.sha256(self.local.read_bytes()).hexdigest()
        b = hashlib.sha256(self.remote.read_bytes()).hexdigest()
        if a != b:
            return False, (f"MISMATCH local {self.local.stat().st_size}B "
                           f"vs durable {self.remote.stat().st_size}B")
        return True, f"identical, {self.local.stat().st_size:,} bytes, sha {a[:12]}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-ephemeral", action="store_true")
    a = ap.parse_args()
    sink = require_durable_sink(allow_ephemeral=a.allow_ephemeral)
    print(f"sink = {sink}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
