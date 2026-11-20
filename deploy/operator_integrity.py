"""Bind packaged code to the runtime package, ELF libraries and CANN kernel objects."""

import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path


PREFIX = Path("/opt/anyinfer/operators")
PACKAGE = PREFIX / "python/vllm_ascend"
VENDOR = PACKAGE / "_cann_ops_custom/vendors/custom_transformer"
MANIFEST = PREFIX / "manifest.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_package():
    spec = importlib.util.find_spec("vllm_ascend")
    if not spec or Path(spec.origin).resolve().parent != PACKAGE:
        raise RuntimeError(f"unexpected vllm_ascend import path: {spec}")


def payload():
    files = {}
    for item in sorted(PACKAGE.rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts:
            files[str(item.relative_to(PACKAGE))] = digest(item)
    return files


def record(soc, source_sha):
    resolve_package()
    if f"soc={soc}" not in (PREFIX / "compiler.txt").read_text().splitlines():
        raise ValueError("compiled SoC differs from release SoC")
    bridge, = PACKAGE.glob("vllm_ascend_C*.so")
    opapi = VENDOR / "op_api/lib/libcust_opapi.so"
    objects = sorted(p for p in (VENDOR / "op_impl").rglob("*") if p.suffix in {".o", ".bin"} and p.is_file())
    libraries = [bridge, opapi]
    if soc.startswith("ascend910"):
        direct, = PACKAGE.glob("libvllm_ascend_kernels.so")
        libraries.append(direct)
    for library in libraries:
        with library.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise ValueError(f"invalid ELF library: {library}")
    if not objects or not any(p.stat().st_size for p in objects):
        raise ValueError("custom OPP has no compiled CANN device kernel objects")
    result = dict(schema=1, soc=soc, source_manifest_sha256=source_sha,
                  wheel_sha256=(PREFIX / "wheel.sha256").read_text().strip(),
                  extension=bridge.name, opapi=str(opapi.relative_to(PACKAGE)),
                  device_kernel_files=len(objects), files=payload())
    MANIFEST.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def verify(expected):
    resolve_package()
    if digest(MANIFEST) != digest(expected):
        raise ValueError("image operator manifest differs from supplied package")
    manifest = json.loads(MANIFEST.read_text())
    if payload() != manifest["files"]:
        raise ValueError("installed operator payload differs from recorded build")
    if os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":")[0] != str(VENDOR):
        raise ValueError("custom OPP search order does not start with the packaged operators")
    import torch
    import torch_npu  # Register CANN runtime libraries before importing the bridge.

    bridge = importlib.import_module("vllm_ascend.vllm_ascend_C")
    extension = Path(bridge.__file__).resolve()
    if extension != PACKAGE / manifest["extension"]:
        raise ValueError(f"loaded unexpected native extension: {extension}")
    opapi = PACKAGE / manifest["opapi"]
    lib = ctypes.CDLL(str(opapi))
    maps = Path("/proc/self/maps").read_text()
    for path in (extension, opapi.resolve()):
        if str(path) not in maps:
            raise ValueError(f"native library not mapped: {path}")
    op = "_C_ascend::npu_hc_post"
    if not torch._C._dispatch_has_kernel_for_dispatch_key(op, "PrivateUse1"):
        raise ValueError(f"no NPU implementation registered for {op}")
    return dict(soc=manifest["soc"], package=str(PACKAGE), extension=str(extension),
                opapi=lib._name, vendor_path=str(VENDOR), verified_npu_registration=op,
                torch_npu_version=torch_npu.__version__, manifest_sha256=digest(MANIFEST),
                device_kernel_files=manifest["device_kernel_files"])


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "verify":
        # Native runtime imports may log to stdout. Reserve the original stream for JSON.
        sys.stdout.flush()
        with os.fdopen(os.dup(sys.stdout.fileno()), "w") as report:
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            try:
                print(json.dumps(verify(Path(sys.argv[2])), indent=2), file=report)
            except Exception as exc:
                print(json.dumps({"status": "failed", "error": str(exc)}), file=report)
                raise
    else:
        raise SystemExit("expected record or verify")
