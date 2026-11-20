#!/usr/bin/env bash
set -euo pipefail
# This entry point is distributed at the root of the editable operator archive.
cd "$(dirname "${BASH_SOURCE[0]}")"
hw=${1:?Usage: ./build-image.sh a2|a3|a5 [linux/arm64|linux/amd64]}
platform=${2:-linux/arm64}
case "$platform" in linux/arm64|linux/amd64) ;; *) echo "Unsupported platform: $platform" >&2; exit 2 ;; esac
case "$hw" in
  a2) soc=ascend910b1; suffix='' ;;
  a3) soc=ascend910_9391; suffix=-a3 ;;
  a5) soc=ascend950dt_9582; suffix=-a5 ;;
  *) echo "Unsupported hardware: $hw" >&2; exit 2 ;;
esac
base=${ANYINFER_BASE_IMAGE:-quay.io/ascend/vllm-ascend:v0.30.0$suffix}
manifest=$(mktemp)
trap 'rm -f -- "$manifest"' EXIT
find operators -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$manifest"
source_hash=$(sha256sum "$manifest" | cut -d' ' -f1)
install -m 644 "$manifest" operator-source.sha256
tag="anyinfer-operators:$hw-${platform#linux/}-${source_hash:0:12}"
flags=()
if [[ -n "${ANYINFER_IIDFILE:-}" ]]; then flags+=(--iidfile "$ANYINFER_IIDFILE"); fi
DOCKER_BUILDKIT=1 docker build --platform "$platform" --pull=false --network=none \
  --build-arg "BASE_IMAGE=$base" --build-arg "ANYINFER_TARGET_SOC=$soc" \
  --build-arg "MAX_JOBS=${ANYINFER_BUILD_JOBS:-8}" --build-arg "OPERATOR_SOURCE_SHA256=$source_hash" \
  "${flags[@]}" -t "$tag" .
echo "Built $tag"
