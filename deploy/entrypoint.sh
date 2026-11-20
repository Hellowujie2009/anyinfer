#!/usr/bin/env bash
# Some CANN environment scripts expect unset variables and a non-strict shell.
for setup in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
  if [[ -r "$setup" ]]; then
    source "$setup" || exit 1
    break
  fi
done
atb=/usr/local/Ascend/nnal/atb/set_env.sh
if [[ -r "$atb" ]]; then source "$atb" || exit 1; fi
set -eu
cd /opt/anyinfer
operator_python=/opt/anyinfer/operators/python
vendor="$operator_python/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export PYTHONPATH="$operator_python${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$vendor/op_api/lib:$operator_python/vllm_ascend${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
case "$ANYINFER_HARDWARE" in a2|a3|a5) ;; *) echo "Invalid hardware family" >&2; exit 1 ;; esac
manifest="/results/build/$ANYINFER_HARDWARE.operators.json"
ASCEND_RT_VISIBLE_DEVICES="$ANYINFER_PROBE_DEVICE" python3 /opt/anyinfer/operators/operator_integrity.py verify \
  "$manifest" > /results/operator-load.json
PYTHONPATH="/app:$PYTHONPATH" ASCEND_RT_VISIBLE_DEVICES="$ANYINFER_PROBE_DEVICE" \
  python3 /opt/anyinfer/operators/smoke_hc_post.py > /results/operator-smoke.json
python3 - <<'PY' > /results/runtime-versions.json
from importlib import metadata
import json
import platform

versions = {"python": platform.python_version()}
for dist in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers"):
    versions[dist] = metadata.version(dist)
print(json.dumps(versions, indent=2))
PY
if [[ ${1:-} == bench ]]; then
  calibration_args=(--devices "${ANYINFER_BENCH_DEVICES:?missing selected devices}"
                    --output "${ANYINFER_CALIBRATION:-/results/hardware-calibration.json}")
  if [[ ${ANYINFER_SKIP_CALIBRATION:-0} == 1 ]]; then calibration_args+=(--skip); fi
  ASCEND_RT_VISIBLE_DEVICES="$ANYINFER_BENCH_DEVICES" python3 /app/calibrate.py "${calibration_args[@]}"
fi
exec /app/bin/anyinfer "$@"
