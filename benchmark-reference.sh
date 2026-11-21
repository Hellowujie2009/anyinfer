#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1:-}" in
  --help|-h)
    cat <<'EOF'
Reference deployment: ./benchmark.sh --hardware a2|a3|a5 --model DIR [--range @survey] [--results DIR] [--golden FILE] [--profile] [--skip-calibration] [-- ENGINE_OPTIONS]
       ./benchmark.sh --list [--range @decode,100-120] [--cases FILE]
Select families, individual IDs or inclusive ranges. Use @survey for a first pass.
Each benchmark shard writes a separate feedback archive under --results (default ./results).
By default each case gets a separate pipeline and memory counter replay. Use -- --aic-metrics pipe or memory for one pass.
See docs/workloads.md and docs/operator-performance.md for workload design and interpretation.
EOF
    exit 0 ;;
  --list)
    shift
    if [[ -d "$root/platforms" ]]; then
      case "$(uname -m)" in
        arm64|aarch64) arch=arm64 ;;
        x86_64) arch=amd64 ;;
        *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
      esac
      root="$root/platforms/$arch"
    fi
    if [[ ! -x "$root/assets/bin/anyinfer" ]]; then
      echo "Run --list from the deployment directory printed by build.sh." >&2
      exit 1
    fi
    export ANYINFER_CASE_CATALOG="$root/assets/benches/cases.json"
    exec "$root/assets/bin/anyinfer" list-cases "$@" ;;
esac
exec bash "$root/run.sh" bench "$@"
