#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run.sh [bench|accuracy|serve|offline] --model HOST_DIR --hardware a2|a3|a5 [options] [-- anyinfer options]
  --devices 0,1,...  Physical davinci device IDs (default all detected IDs)
  --range SELECTOR  Workload families (e.g. @survey), inclusive ranges or individual IDs; default all
  --results DIR     Host parent directory for independent run folders (default ./results)
  --cases FILE      Use an external benchmark JSON file
  --golden FILE     Check reference generations before running the benchmark
  --profile         Extend per-case NPU operator profiling beyond the default short replay
  --skip-calibration  Skip device reference measurements before model loading
  --job-dir DIR     Read-only input directory mounted at /job
  --shm-size SIZE   Docker shared memory limit (default 512g)
EOF
}
die() { echo "run: $*" >&2; exit 1; }
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "$(uname -m)" in aarch64|arm64) arch=arm64 ;; x86_64) arch=amd64 ;; *) die "unsupported CPU architecture" ;; esac
if [[ -d "$bundle/platforms" ]]; then
  exec "$bundle/platforms/$arch/run.sh" "$@"
fi
mode=bench
case "${1:-}" in bench|accuracy|serve|offline) mode=$1; shift ;; esac
model='' hw='' devices='' job='' cases='' golden='' profile=0 skip_calibration=0
results="$PWD/results" range=all shm=512g
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model) model=${2:?missing model directory}; shift 2 ;;
    --hardware) hw=${2:?missing hardware family}; shift 2 ;;
    --devices) devices=${2:?missing device list}; shift 2 ;;
    --results) results=${2:?missing output directory}; shift 2 ;;
    --range) range=${2:?missing case range}; shift 2 ;;
    --profile) profile=1; shift ;;
    --skip-calibration) skip_calibration=1; shift ;;
    --shm-size) shm=${2:?missing size}; shift 2 ;;
    --cases) cases=${2:?missing cases file}; shift 2 ;;
    --golden) golden=${2:?missing golden JSONL file}; shift 2 ;;
    --job-dir) job=${2:?missing input directory}; shift 2 ;;
    --) shift; break ;;
    *) die "unknown option $1 (pass binary flags after --)" ;;
  esac
done
[[ "$hw" =~ ^a[235]$ ]] || die "select --hardware a2, a3 or a5"
mkdir -p "$results"
results=$(cd "$results" && pwd -P)
run=$(mktemp -d "$results/$mode-$hw-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX")
mkdir "$run/telemetry"
echo "Results: $run" >&2
start=$(date -u +%FT%TZ)
sampler=''
write_status() {
  printf '{"started_utc":"%s","ended_utc":"%s","exit_code":%s,"hardware":"%s","cpu_arch":"%s"}\n' \
    "$start" "$(date -u +%FT%TZ)" "$1" "$hw" "$arch" > "$run/status.json"
}
write_feedback() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -B "$bundle/tools/report.py" --feedback "$run"
  elif [[ -n "${started_container:-}" ]]; then
    docker run --rm --pull=never --network none --entrypoint python3 \
      -v "$bundle/tools:/tools:ro" -v "$run:/input/${run##*/}:ro" "$id" \
      -B /tools/report.py --feedback "/input/${run##*/}"
  else
    # Even a machine that fails preflight before loading the runtime can return diagnostics.
    printf '{"ok":false,"error":"Host preflight failed. See status.json and stderr.log."}\n'
  fi
}
finish() {
  rc=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$sampler" ]]; then
    kill "$sampler" 2>/dev/null
    wait "$sampler" 2>/dev/null
    if [[ "$mode" == bench ]] && ! awk -F '\t' 'NR>1 && $3 == "0" {ok=1} END{exit !ok}' "$run/telemetry/polls.tsv"; then
      echo "Device sampling produced no successful periodic npu-smi readings" >&2
      if ((rc == 0)); then rc=1; fi
    fi
  fi
  if [[ -n "${docker_pid:-}" ]]; then
    kill "$docker_pid" 2>/dev/null
  fi
  if [[ -s "$run/container.id" ]]; then
    docker rm -f "$(cat "$run/container.id")" >/dev/null 2>&1
  fi
  if [[ -n "${docker_pid:-}" ]]; then wait "$docker_pid"; fi
  if [[ -n "${stdout_logger:-}" ]]; then wait "$stdout_logger"; fi
  if [[ -n "${started_container:-}" && "$(id -u)" != 0 ]]; then
    docker run --rm --network none --user 0 -v "$run:/results" --entrypoint /bin/chown \
      "$id" -Rh "$(id -u):$(id -g)" /results || echo "Could not change output file ownership" >&2
  fi
  if [[ -n "${stderr_logger:-}" ]]; then
    exec 2>&3 3>&-
    wait "$stderr_logger"
  fi
  write_status "$rc"
  if [[ "$mode" == bench ]]; then
    if write_feedback > "$run/feedback.json.part" 2>>"$run/stderr.log" && mv "$run/feedback.json.part" "$run/feedback.json"; then
      echo "Benchmark summary: $run/feedback.json" >&2
    else
      rm -f -- "$run/feedback.json.part"
      echo "Could not summarize the run; raw results are preserved" >&2
      if ((rc == 0)); then rc=1; fi
      write_status "$rc"
      printf '{"ok":false,"error":"Failed to summarize results. See stderr.log and raw case files."}\n' > "$run/feedback.json"
    fi
    if tar -C "$results" -cf - "${run##*/}" | gzip -1 > "$run.tar.gz.part" && mv "$run.tar.gz.part" "$run.tar.gz"; then
      echo "Feedback archive: $run.tar.gz" >&2
      (cd "$results" && sha256sum "${run##*/}.tar.gz" > "${run##*/}.tar.gz.sha256") || rc=1
    else
      rm -f -- "$run.tar.gz.part"
      echo "Result archive failed; raw files are at $run" >&2
      if ((rc == 0)); then rc=1; fi
    fi
  fi
  echo "Exit $rc. Results: $run" >&2
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec 3>&2
exec 2> >(tee -a "$run/stderr.log" >&3)
stderr_logger=$!
if [[ "$mode" == bench ]]; then
  printf 'hardware=%s\nrange=%s\nmodel=%q\nhost=%q\n' "$hw" "$range" "$model" "$(hostname)" > "$run/selection.txt"
fi
[[ -f "$bundle/build/$hw.id" ]] && grep -Fq "  build/$hw.id" "$bundle/build/payload.sha256" || die "this bundle does not include $hw"
[[ -d "$model" ]] || die "provide a local model directory with --model"
model=$(cd "$model" && pwd -P)
[[ -r "$model/config.json" ]] || die "checkpoint config.json missing at $model"
[[ "$profile" == 0 || "$mode" == bench ]] || die "--profile applies to benchmark cases"
[[ "$skip_calibration" == 0 || "$mode" == bench ]] || die "--skip-calibration applies to benchmark cases"
[[ "$mode" == bench || ( "$range" == all && -z "$cases" ) ]] || die "case selection applies to bench"
if [[ -n "$golden" ]]; then
  [[ "$mode" == bench && -f "$golden" && -s "$golden" && -r "$golden" ]] || die "--golden needs a readable, nonempty host JSONL file in bench mode"
fi
if [[ "$mode" == bench ]]; then
  cp -- "${cases:-$bundle/assets/benches/cases.json}" "$run/cases.json"
  if [[ -n "$golden" ]]; then
    cp -- "$golden" "$run/golden.jsonl"
    (cd "$run" && sha256sum golden.jsonl > golden.sha256)
  fi
fi
cp -r "$bundle/build" "$run/build"
(cd "$bundle" && sha256sum --quiet -c build/payload.sha256)
cp "$bundle/assets/operator-index.json" "$run/build/operator-index.json"
grep -Fxq "platform=linux/$arch" "$run/build/build.txt" || die "CPU architecture does not match this bundle"
if [[ -z "$devices" ]]; then
  shopt -s nullglob
  nodes=(/dev/davinci[0-9]*)
  (("${#nodes[@]}")) || die "no Ascend /dev/davinciN device nodes"
  devices=$(printf '%s\n' "${nodes[@]#/dev/davinci}" | sort -n | paste -sd,)
fi
[[ "$devices" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || die "invalid device list: $devices"
IFS=, read -r -a selected <<< "$devices"
args=(--rm --init --cidfile "$run/container.id" --network host "--shm-size=$shm"
  --ulimit memlock=-1:-1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e ANYINFER_OPERATOR_INDEX=/app/operator-index.json -e "ANYINFER_HARDWARE=$hw"
  -e "ANYINFER_PROBE_DEVICE=${selected[0]}"
  --volume "$bundle/assets:/app:ro" --volume "$model:/model:ro" --volume "$run:/results"
  --entrypoint /bin/bash)
if [[ "$mode" == bench ]]; then
  args+=(-e "ANYINFER_BENCH_DEVICES=$devices" -e "ANYINFER_SKIP_CALIBRATION=$skip_calibration"
         -e ANYINFER_CALIBRATION=/results/hardware-calibration.json
         -e ANYINFER_HC_BENCH_CONFIG=/model/config.json)
fi
if grep -q 'executable=False$' "$bundle/build/$hw.frontend.txt" 2>/dev/null; then
  args+=(-e ANYINFER_FRONTEND=python)
  echo "This image has no executable vllm-rs. Defaulting to the packaged Python frontend; --frontend overrides this choice." >&2
fi
nodes=(/dev/davinci_manager /dev/hisi_hdc)
for n in "${selected[@]}"; do nodes+=("/dev/davinci$n"); done
if [[ "$hw" == a5 ]]; then
  nodes+=(/dev/ummu /dev/uburma)
  mounts=(/usr/local/Ascend/driver /usr/local/sbin)
  for path in /etc/hccl_rootinfo.json /lib/route.conf /usr/bin/urma_admin; do
    if [[ -e "$path" ]]; then mounts+=("$path"); fi
  done
else
  nodes+=(/dev/devmm_svm)
  mounts=(/usr/local/Ascend/driver/lib64 /usr/local/Ascend/driver/version.info)
  for path in /usr/local/Ascend/driver/tools/hccn_tool /etc/hccn.conf; do
    if [[ -e "$path" ]]; then mounts+=("$path"); fi
  done
  args+=(--privileged)
fi
mounts+=(/etc/ascend_install.info /usr/local/dcmi)
for node in "${nodes[@]}"; do
  [[ -e "$node" ]] || die "missing device: $node"
  args+=(--device "$node")
done
for path in "${mounts[@]}"; do
  [[ -e "$path" ]] || die "missing driver component: $path"
  args+=(--volume "$path:$path:ro")
done
smi=$(command -v npu-smi || true)
[[ -x "$smi" ]] || die "npu-smi is required on the target host"
args+=(--volume "$(realpath "$smi"):/usr/local/bin/npu-smi:ro")
if [[ -n "$job" ]]; then
  job=$(cd "$job" && pwd -P)
  args+=(--volume "$job:/job:ro")
fi
command_args=("$mode" --model /model --hardware "$hw" --devices "$devices")
if [[ "$mode" == bench ]]; then
  command_args+=(--range "$range" --cases /results/cases.json --output /results/bench)
  if [[ -n "$golden" ]]; then
    command_args+=(--golden /results/golden.jsonl)
  fi
  if ((profile)); then
    command_args+=(--profile)
  fi
elif [[ "$mode" == offline ]]; then
  args+=(-i)
fi
command_args+=("$@")
printf '%q ' "${command_args[@]}" > "$run/command.txt"
printf '\n' >> "$run/command.txt"
printf 'host=%s\nmodel=%s\ndevices=%s\nshm=%s\nprofile=%s\n' "$(hostname)" "$model" "$devices" "$shm" "$profile" > "$run/host.txt"
{
  uname -a
  lscpu
  cat /proc/meminfo
  cat /usr/local/Ascend/driver/version.info
  docker version
} >> "$run/host.txt" 2>&1 || true
(cd "$model" && sha256sum config.json && for path in *.index.json tokenizer.json; do
  if [[ -f "$path" ]]; then sha256sum "$path"; fi
done) > "$run/checkpoint-metadata.sha256"
(cd "$model" && find -L . -type f -name '*.safetensors' -printf '%P\t%s\t%T@\n' | LC_ALL=C sort) > "$run/weight-inventory.tsv"
if [[ -f "$model/anyinfer-weights.sha256" ]]; then
  cp "$model/anyinfer-weights.sha256" "$run/checkpoint-supplied-weights.sha256"
fi
timeout -k 2 15 "$smi" info > "$run/telemetry/initial.txt" || die "npu-smi info failed"
for field in topo mapping; do
  query=(-t topo)
  if [[ "$field" == mapping ]]; then query=(-m); fi
  timeout -k 2 15 "$smi" info "${query[@]}" > "$run/telemetry/$field.txt" 2>&1 ||
    printf '\nquery exit=%d\n' "$?" >> "$run/telemetry/$field.txt"
done
id=$(cat "$bundle/build/$hw.ref")
if ! docker image inspect "$id" >/dev/null 2>&1; then
  docker image load -i "$bundle/images/$hw.tar" >&2
fi
docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$id" > "$run/image.txt"
docker image inspect --format '{{json .RootFS.Layers}}' "$id" | cmp -s - "$bundle/build/$hw.layers.json" ||
  die "cached image layers differ from the packaged $hw image; reload images/$hw.tar"
sample() {
  local timestamp probe_rc
  while :; do
    timestamp=$(date -u +%s%3N)
    probe_rc=0
    {
      printf '\n[%s]\n' "$(date -u +%FT%T.%NZ)"
      timeout -k 2 10 "$smi" info || probe_rc=$?
      if ((probe_rc)); then echo "npu-smi exit=$probe_rc"; fi
      head -n 1 /proc/stat
      cat /proc/loadavg
      awk '/^(MemAvailable|SwapFree):/' /proc/meminfo
    } >> "$run/telemetry/samples.log" 2>&1
    printf '%s\t%s\t%s\n' "$timestamp" "$(date -u +%s%3N)" "$probe_rc" >> "$run/telemetry/polls.tsv"
    sleep 2
  done
}
printf 'start_unix_ms\tend_unix_ms\tnpu_smi_exit\n' > "$run/telemetry/polls.tsv"
sample & sampler=$!
started_container=1
# A shell waiting on an asynchronous child can run the signal trap immediately.
# Preserve stdin for offline JSONL and wait for the log writer before archiving.
exec 4<&0 5> >(tee "$run/stdout.log")
stdout_logger=$!
docker run --pull=never "${args[@]}" "$id" /app/entrypoint.sh "${command_args[@]}" <&4 >&5 &
docker_pid=$!
exec 4<&- 5>&-
rc=0
wait "$docker_pid" || rc=$?
docker_pid=''
wait "$stdout_logger" || rc=1
stdout_logger=''
exit "$rc"
