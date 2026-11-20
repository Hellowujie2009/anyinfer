"""Prepare an editable vLLM-Ascend source copy for an offline CANN build."""

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def prepare(source):
    here = Path(__file__).resolve().parent
    archives = here / "build-deps"
    third_party = source / "csrc" / "third_party"
    if not (third_party / "catlass/include/catlass").is_dir():
        raise ValueError(f"missing Catlass headers in {source}")

    for item in json.loads((here / "SOURCE.json").read_text())["build_dependencies"]:
        path = archives / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"archive checksum mismatch: {path}")

    # The two ExternalProject targets apply patches while building the source.
    packages = third_party / "pkg"
    packages.mkdir(exist_ok=True)
    for pkg in ("abseil-cpp-20230802.1.tar.gz", "protobuf-25.1.tar.gz"):
        shutil.copyfile(archives / pkg, packages / pkg)

    with zipfile.ZipFile(archives / "include.zip") as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if path.parts[:1] != ("include",) or member.is_dir():
                continue
            if ".." in path.parts:
                raise ValueError(f"invalid header path: {path}")
            dest = third_party / "json" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(bundle.read(member))

    makeself = third_party / "makeself"
    makeself.mkdir(exist_ok=True)
    with tarfile.open(archives / "makeself-release-2.5.0-patch1.tar.gz") as bundle:
        for name in ("makeself.sh", "makeself-header.sh"):
            dest = makeself / name
            with bundle.extractfile("makeself-release-2.5.0/" + name) as script:
                dest.write_bytes(script.read())
            dest.chmod(0o755)

    # Windows checkouts and Docker build contexts can lose executable bits.
    for name in (here / "executable-files.txt").read_text().splitlines():
        path = source / name
        path.chmod(path.stat().st_mode | 0o111)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="mutable vllm-ascend build tree")
    prepare(parser.parse_args().source.resolve())
