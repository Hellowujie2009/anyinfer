"""Publish a finished build on the output filesystem and retain its predecessor."""

import ctypes
import errno
import fcntl
import os
import signal
import sys
from pathlib import Path


def replace_release(candidate, current, previous):
    # Keep both trees outside the temporary staging directory during the swap.
    # Even an interrupted publisher must leave the previous tree recoverable.
    exchange = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if exchange is not None:
        candidate.rename(previous)
        # AT_FDCWD, RENAME_EXCHANGE. Both paths are real directories on one filesystem.
        if exchange(-100, os.fsencode(previous), -100, os.fsencode(current), 2) == 0:
            return
        error = ctypes.get_errno()
        previous.rename(candidate)
        if error not in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
            raise OSError(error, os.strerror(error))

    # Some network filesystems lack RENAME_EXCHANGE. Keep the old directory
    # under its final backup name throughout the replacement.
    signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    try:
        current.rename(previous)
        try:
            candidate.rename(current)
        except BaseException:
            previous.rename(current)
            raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def publish(stage, name):
    out = stage.parent
    current = out / "anyinfer"
    previous = out / ("previous-" + name.removesuffix(".tar.gz"))
    candidate = stage / "anyinfer"
    # Serialize only publication. Separate builds can compile concurrently.
    lock = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exists = os.path.lexists(current)
        if current.is_symlink() or (exists and not current.is_dir()):
            raise ValueError(f"expected a real release directory: {current}")
        marker = current / "release-format.txt"
        if exists and (marker.is_symlink() or not marker.is_file()):
            raise ValueError(f"existing directory has no AnyInfer release marker: {current}; choose another --out")
        if exists and marker.read_bytes() != b"anyinfer-release-v1\n":
            raise ValueError(f"unrecognized release directory: {current}; choose another --out")
        if exists and current.resolve() == Path.cwd().resolve():
            raise ValueError("the output would replace the source directory; choose another --out")
        artifacts = [name, name + ".sha256"]
        if any(os.path.lexists(p) for p in [previous, *(out / n for n in artifacts)]):
            raise FileExistsError(f"release name collision: {name}")
        for artifact in artifacts:
            (stage / artifact).rename(out / artifact)
        if exists:
            replace_release(candidate, current, previous)
            print(f"Previous deployment retained: {previous}")
        else:
            candidate.rename(current)
    finally:
        os.close(lock)


if __name__ == "__main__":
    publish(Path(sys.argv[1]).resolve(), sys.argv[2])
