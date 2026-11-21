#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  cat <<'EOF'
./benchmark.sh --list [--range 1,8-24]
./benchmark.sh --host-only [--results DIR]
./benchmark.sh [--soc CANN_SOC] [--device 0] [--range all] [--results DIR]
               [--profile both|memory|pipe|off] [--samples N] [--iterations N] [--warmup N]
               [--tile 128|256|512|1024]
Each invocation tests one device and emits its own feedback archive. A package
with one compiled SoC selects it automatically. The default runs all native
operator cases and separately captures pipeline and memory counters.
Host-only packages default to the CPU scheduler suite. A device package can
also run this suite with --host-only, without initializing CANN.
EOF
  exit 0
fi
[[ -x "$root/bin/anyinfer" ]] || { echo "Use benchmark.sh inside the built deployment directory." >&2; exit 1; }
soc='' results="$PWD/results" args=() host_only=0
if [[ -f "$root/release-format.txt" && $(cat "$root/release-format.txt") == anyinfer-host-release-v1 ]]; then host_only=1; fi
if [[ ${1:-} == --list ]]; then
  if ((host_only)); then
    echo "This host package runs the scheduler suite together. Use ./benchmark.sh [--results DIR]."
    exit 0
  fi
  shift
  exec "$root/bin/anyinfer" list-cases "$@"
fi
while (($#)); do
  case "$1" in
    --soc) soc=${2:?missing SoC name}; shift 2 ;;
    --results) results=${2:?missing results directory}; shift 2 ;;
    --host-only) host_only=1; shift ;;
    --range|--device|--profile|--samples|--iterations|--warmup|--tile) args+=("$1" "${2:?missing value}"); shift 2 ;;
    *) echo "Unknown benchmark option: $1" >&2; exit 2 ;;
  esac
done
if ((host_only)) && [[ -n "$soc" || ${#args[@]} -gt 0 ]]; then
  echo "Device workload options do not apply to --host-only." >&2
  exit 2
fi
if ((!host_only)) && [[ -z "$soc" ]]; then
  shopt -s nullglob
  candidates=("$root"/kernels/*/bin/anyinfer-hc-post-bench)
  (( ${#candidates[@]} == 1 )) || { echo "Select an installed --soc for this device." >&2; exit 2; }
  soc=$(basename "$(dirname "$(dirname "${candidates[0]}")")")
fi
label=host-scheduler
soc_json=null
if ((!host_only)); then
  [[ "$soc" =~ ^[A-Za-z0-9_]+$ && -x "$root/kernels/$soc/bin/anyinfer-hc-post-bench" ]] ||
    { echo "Missing installed kernel for SoC $soc" >&2; exit 2; }
  if [[ -z ${ASCEND_HOME_PATH:-} ]]; then
    for setup in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
      if [[ -r "$setup" ]]; then set +u; source "$setup"; set -u; break; fi
    done
  fi
  export LD_LIBRARY_PATH="$root/kernels/$soc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  label=native-$soc
  soc_json="\"$soc\""
fi
mkdir -p -- "$results"
results=$(cd "$results" && pwd -P)
run=$(mktemp -d "$results/bench-$label-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX")
name=${run##*/}
finish() {
  local rc=$?
  trap - EXIT INT TERM
  printf '{"exit_code":%d,"soc":%s,"ended_utc":"%s"}\n' "$rc" "$soc_json" "$(date -u +%FT%TZ)" > "$run/status.json"
  tar -C "$results" -cf - "$name" | gzip -1 > "$run.tar.gz.part" && mv "$run.tar.gz.part" "$run.tar.gz" ||
    { echo "Archive failed. Results directory: $run" >&2; exit 1; }
  (cd "$results" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256") || exit 1
  printf 'Feedback archive: %s\n' "$run.tar.gz"
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp -a "$root/build" "$run/build"
printf '%q scheduler-bench\n' "$root/bin/anyinfer" > "$run/command.txt"
(cd "$root" && sha256sum --quiet -c build/payload.sha256)
"$root/bin/anyinfer" scheduler-bench > "$run/host-scheduler.json.part" 2>"$run/host-scheduler.stderr.log"
mv "$run/host-scheduler.json.part" "$run/host-scheduler.json"
if ((host_only)); then exit 0; fi
printf '%q ' "$root/bin/anyinfer" bench --worker "$root/kernels/$soc/bin/anyinfer-hc-post-bench" --output "$run/benchmark" "${args[@]}" >> "$run/command.txt"
printf '\n' >> "$run/command.txt"
"$root/bin/anyinfer" bench --worker "$root/kernels/$soc/bin/anyinfer-hc-post-bench" \
  --output "$run/benchmark" "${args[@]}" > "$run/stdout.log" 2> >(tee "$run/stderr.log" >&2)
