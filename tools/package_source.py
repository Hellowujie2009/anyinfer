"""Pack the editable AnyInfer project for a Linux build host, from Windows or Linux."""

import argparse
import gzip
import hashlib
import os
import re
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUTS = (
    ".gitattributes", ".gitignore", "Cargo.toml", "Cargo.lock", "README.md",
    "build.sh", "benchmark.sh", "run-reference.sh", "src", "benches", "tests",
    "reference", "native", "tools", "docs", "deploy", "operators", "build-reference.sh", "benchmark-reference.sh", "README-reference.md",
)


def package(out):
    inputs = list(INPUTS)
    if (ROOT / "backend-overlay").is_dir():
        inputs.append("backend-overlay")
    files = []
    for name in inputs:
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = path.rglob("*") if path.is_dir() else [path]
        files.extend(p.relative_to(ROOT) for p in candidates if p.is_file()
                     and "__pycache__" not in p.parts and "target" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    executables = set((ROOT / "operators/executable-files.txt").read_text(encoding="utf-8").splitlines())
    version = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M)[1]
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out, suffix=".part", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|") as archive:
                for relative in sorted(files, key=Path.as_posix):
                    source = ROOT / relative
                    name = relative.as_posix()
                    entry = tarfile.TarInfo("anyinfer/" + name)
                    entry.size = source.stat().st_size
                    is_executable = (source.suffix == ".sh"
                                     or name.removeprefix("operators/vllm-ascend/") in executables)
                    entry.mode = 0o755 if is_executable else 0o644
                    with source.open("rb") as stream:
                        archive.addfile(entry, stream)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        destination = out / f"anyinfer-source-{version}-{checksum[:12]}.tar.gz"
        path.chmod(0o644)
        os.replace(path, destination)
        destination.with_suffix(".gz.sha256").write_bytes(f"{checksum}  {destination.name}\n".encode("ascii"))
        return destination
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    print(package(parser.parse_args().out.resolve()))
