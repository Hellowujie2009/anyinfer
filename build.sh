#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out="$root/dist"
socs=()
sources=0
host_only=0
while (($#)); do
  case "$1" in
    --soc) socs+=("${2:?missing CANN SoC version}"); shift 2 ;;
    --out) out=${2:?missing output directory}; shift 2 ;;
    --sources-only) sources=1; shift ;;
    --host-only) host_only=1; shift ;;
    -h|--help) cat <<'EOF'
bash build.sh --soc Ascend910B1 [--soc ANOTHER_CANN_SOC] [--out DIR]
bash build.sh --sources-only [--out DIR]
bash build.sh --host-only [--out DIR]
Build the standalone Rust tools and native V4 Ascend kernel benchmark with the
installed CANN SDK. DIR/anyinfer is the deployable directory. Use the exact
CANN SoC version of your target. build-reference.sh builds the vLLM baseline.
The host-only build uses Cargo and ships DIR/anyinfer-host for CPU scheduler tests.
EOF
      exit 0 ;;
    *) echo "Unknown build option: $1" >&2; exit 2 ;;
  esac
done
if ((host_only && (sources || ${#socs[@]}))); then
  echo "--host-only cannot be combined with --soc or --sources-only" >&2
  exit 2
fi
name=anyinfer
if ((host_only)); then name=anyinfer-host; fi
mkdir -p -- "$out"
out=$(cd "$out" && pwd -P)
stage=$(mktemp -d "$out/.anyinfer-native-XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
bundle="$stage/$name"
mkdir -p "$bundle"/{src/native,docs,native,tests}
cd "$root"
install -m 644 Cargo.toml Cargo.lock README.md "$bundle/"
install -m 755 build.sh benchmark.sh "$bundle/"
install -m 644 docs/user-manual.md docs/native-model.md "$bundle/docs/"
install -m 644 src/{lib,main,opbench,scheduler_bench,host_observation}.rs "$bundle/src/"
install -m 644 tests/host_decode_allocations.rs "$bundle/tests/"
install -m 644 src/native/*.rs "$bundle/src/native/"
cp -a native/. "$bundle/native/"
(cd "$bundle" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$stage/source.sha256")
source_sha=$(sha256sum "$stage/source.sha256" | cut -c1-12)
version=$(awk -F'"' '/^version = /{print $2;exit}' Cargo.toml)
if (( sources )); then
  archive="anyinfer-native-source-$version-$source_sha.tar.gz"
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$stage" -cf - anyinfer | gzip -n > "$stage/$archive"
  mv -f "$stage/$archive" "$out/$archive"
  (cd "$out" && sha256sum "$archive" > "$archive.sha256")
  echo "$out/$archive"
  exit 0
fi
((host_only || ${#socs[@]})) || { echo "Select --host-only or specify the target's CANN --soc (e.g. Ascend910B1)." >&2; exit 2; }
for soc in "${socs[@]}"; do
  [[ "$soc" =~ ^[A-Za-z0-9_]+$ ]] || { echo "Invalid SoC version: $soc" >&2; exit 2; }
done
setup_cann() {
  if [[ -n ${ASCEND_HOME_PATH:-} ]]; then return; fi
  local script
  for script in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
    if [[ -r "$script" ]]; then
      set +u
      source "$script"
      set -u
      return
    fi
  done
}
if ((!host_only)); then
  setup_cann
  : "${ASCEND_HOME_PATH:?source your CANN toolkit set_env.sh first}"
fi
mkdir -p "$bundle"/{bin,build,kernels}
install -m 644 "$stage/source.sha256" "$bundle/build/source.sha256"
# Source archives normalize timestamps. Key the Cargo output by content so a
# previously unpacked version cannot be reused after overwriting a source tree.
target_dir="${CARGO_TARGET_DIR:-$root/target}/anyinfer-native-$source_sha"
mkdir -p "$target_dir"
target_dir=$(cd "$target_dir" && pwd -P)
cd "$bundle"
cargo test --locked --release --lib --bin anyinfer --test host_decode_allocations --target-dir "$target_dir"
cargo build --locked --release --bin anyinfer --target-dir "$target_dir"
install -m 755 "$target_dir/release/anyinfer" "$bundle/bin/"
if ((!host_only)); then
  cmake -S native/head_runner -B "$stage/head-runner" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$bundle"
  cmake --build "$stage/head-runner" --parallel
  cmake --install "$stage/head-runner"
  "$bundle/bin/anyinfer" list-cases > "$bundle/build/cases.json"
fi
{
  printf 'source=%s\nutc=%s\ncpu=%s\nsocs=%s\n' "$source_sha" "$(date -u +%FT%TZ)" "$(uname -m)" "${socs[*]}"
  rustc -vV
  if ((!host_only)); then
    cmake --version | head -n 1
    for f in "$ASCEND_HOME_PATH/version.info" "$ASCEND_HOME_PATH/../version.info"; do
      if [[ -f "$f" ]]; then cat "$f"; break; fi
    done
  fi
} > "$bundle/build/build.txt"
for soc in "${socs[@]}"; do
  prefix="$bundle/kernels/$soc"
  cmake -S native/acl_backend -B "$stage/cmake-$soc" -DSOC_VERSION="$soc" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix"
  cmake --build "$stage/cmake-$soc" --parallel
  ctest --test-dir "$stage/cmake-$soc" --output-on-failure
  cmake --install "$stage/cmake-$soc"
  test -x "$prefix/bin/anyinfer-hc-post-bench"
done
kind=native
if ((host_only)); then kind=host; fi
format="anyinfer-$kind-release-v1"
printf '%s\n' "$format" > "$bundle/release-format.txt"
(cd "$bundle" && find bin kernels -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > build/payload.sha256)
archive="anyinfer-$kind-$version-$(uname -m)-$source_sha-${stage##*-}.tar.gz"
tar -C "$stage" -cf - "$name" | gzip -1 > "$stage/$archive"
(cd "$stage" && sha256sum "$archive" > "$archive.sha256")
# Serialize publication if several builds target the same output directory.
exec 9>"$out/.native-publish.lock"
flock 9
dest="$out/$name"
if [[ -e "$dest" || -L "$dest" ]]; then
  [[ ! -L "$dest" && -f "$dest/release-format.txt" &&
     $(cat "$dest/release-format.txt") == "$format" ]] ||
    { echo "$dest already exists with a different format; select --out DIR" >&2; exit 1; }
  mv "$dest" "$stage/previous"
fi
if ! mv "$bundle" "$dest"; then
  if [[ -d "$stage/previous" ]]; then mv "$stage/previous" "$dest"; fi
  exit 1
fi
mv "$stage/$archive" "$stage/$archive.sha256" "$out/"
printf 'Deployment: %s\nBenchmark: %s\nPackage: %s\n' "$dest" "$dest/benchmark.sh" "$out/$archive"
