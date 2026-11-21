"""Check the local Ascend environment, then replace this process with vLLM."""

import os
import sys

# The offline CLI reserves stdout for JSONL. This also redirects C++/HCCL logs.
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

from importlib.metadata import version

from packaging.version import Version
import torch_npu
from vllm_ascend.device.device_config import check_ascend_device_type, get_ascend_device_type

hardware, device_count, *serve_args = sys.argv[1:]
core = version("vllm")
plugin = version("vllm-ascend")
print(f"vLLM {core}, vLLM-Ascend {plugin}, torch-npu {version('torch-npu')}", flush=True)
# An official image built from an untagged source snapshot can report 0.0.0
# for the plugin. The release manifest records the image ID in this case.
if Version(core).base_version != "0.30.0" or Version(plugin).base_version not in {"0.30.0", "0.0.0"}:
    raise RuntimeError("anyinfer requires the vLLM-Ascend 0.30.0 release image")
check_ascend_device_type()
if get_ascend_device_type().name.lower() != hardware:
    raise RuntimeError(f"selected {hardware}, installed runtime targets {get_ascend_device_type().name}")
if torch_npu.npu.device_count() != int(device_count):
    raise RuntimeError("visible NPU count does not match --devices; check container device mounts")

# Exec clears the probe's NPU context before spawning TP/DP ranks.
os.execv(sys.executable, [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *serve_args])
