#!/usr/bin/env bash
# Build inside the matching CANN/PyTorch image (no local NPU required).
for setup in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
  if [[ -r "$setup" ]]; then source "$setup" || exit 1; break; fi
done
if [[ -r /usr/local/Ascend/nnal/atb/set_env.sh ]]; then source /usr/local/Ascend/nnal/atb/set_env.sh || exit 1; fi
set -euo pipefail
export SOC_VERSION="${ANYINFER_TARGET_SOC:?target SoC is required}" COMPILE_CUSTOM_KERNELS=1 VLLM_BATCH_INVARIANT=0
export SETUPTOOLS_SCM_PRETEND_VERSION=0.30.0 CMAKE_BUILD_TYPE=Release
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
source_root=/opt/anyinfer-build/operators/vllm-ascend
destination=/opt/anyinfer/operators
sha256sum --quiet -c operator-source.sha256
if [[ ! -d "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}" ]]; then
  echo "CANN compiler toolkit missing from the selected build image" >&2
  exit 1
fi
python3 operators/prepare_build_deps.py "$source_root"
cd "$source_root"
# Compiler outputs must originate in this build. Never seed a previous wheel's
# bridge, CMake cache, OPP or kernel binaries into the compiler worktree.
for path in build csrc/build csrc/output csrc/build_out; do
  if [[ -e "$path" ]]; then echo "Unexpected build output in operator sources: $path" >&2; exit 1; fi
done
if [[ -d vllm_ascend/_cann_ops_custom/vendors || -n "$(find vllm_ascend -name '*.so' -print -quit)" ]]; then
  echo "Unexpected precompiled vendor package or extension in operator sources" >&2
  exit 1
fi
python3 -m pip wheel --no-cache-dir --no-deps --no-build-isolation --no-index -w /tmp/anyinfer-wheels .
wheels=(/tmp/anyinfer-wheels/vllm_ascend-*.whl)
[[ ${#wheels[@]} == 1 && -f "${wheels[0]}" ]]
mkdir -p "$destination/python"
python3 -m pip install --no-index --no-cache-dir --no-compile --no-deps --target "$destination/python" "${wheels[0]}"
# Preserve JIT sources, configs and other package data omitted by setuptools.
cp -an vllm_ascend/. "$destination/python/vllm_ascend/"
sha256sum "${wheels[0]}" | awk '{print $1}' > "$destination/wheel.sha256"
{
  printf 'soc=%s\n' "$SOC_VERSION"
  cmake --version
  c++ --version
  python3 -m pip show torch torch-npu pybind11 setuptools-scm
} > "$destination/compiler.txt"
