#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash build-reference.sh [--profiles a2,a3,a5] [--platform linux/arm64|linux/amd64|all] [--out DIR] [--local-images] [--sources-only]
Build and test the Rust binary and Ascend kernels, then publish DIR/anyinfer/ and a matching archive.
The published directory contains benchmark.sh, runtime images, cases, sources and docs/reference-manual.md.
Defaults: all hardware profiles, the build host CPU architecture, dist/. Requires Linux and Docker.
EOF
}
die() { echo "build: $*" >&2; exit 1; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profiles=a2,a3,a5
platform=''
local_images=0
sources_only=0
out="$root/dist/reference"
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --profiles) profiles=${2:?missing profile list}; shift 2 ;;
    --platform) platform=${2:?missing platform}; shift 2 ;;
    --out) out=${2:?missing output directory}; shift 2 ;;
    --local-images) local_images=1; shift ;;
    --sources-only) sources_only=1; shift ;;
    *) die "unknown option $1" ;;
  esac
done
if [[ -z "$platform" ]]; then
  case "$(uname -m)" in
    arm64|aarch64) platform=linux/arm64 ;;
    x86_64) platform=linux/amd64 ;;
    *) die "unsupported host architecture; specify --platform" ;;
  esac
fi
[[ "$platform" == linux/arm64 || "$platform" == linux/amd64 || "$platform" == all ]] || die "unsupported platform: $platform"
[[ "$profiles" =~ ^(a[235])(,a[235])*$ ]] || die "expected comma-separated a2, a3, a5"
archive_arch=${platform#linux/}
builder=${ANYINFER_BUILDER_IMAGE:-rust:1.100.0-alpine3.23}
mkdir -p "$out"
out=$(cd "$out" && pwd)
stage=$(mktemp -d "$out/.build-XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s tests
work_context="$stage/anyinfer-operators"
mkdir -p "$work_context/overlay/files"
cp -a operators "$work_context/operators"
install -m 644 deploy/operators.Dockerfile "$work_context/Dockerfile"
install -m 644 deploy/compile-operators.sh deploy/operator_integrity.py deploy/calibrate.py "$work_context/"
install -m 755 deploy/build-operator-image.sh "$work_context/build-image.sh"
install -m 644 deploy/apply_overlay.py "$work_context/overlay/"
(cd "$work_context" && find operators -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > operator-source.sha256)
operator_source_sha=$(sha256sum "$work_context/operator-source.sha256" | cut -d' ' -f1)
python3 tools/operator_catalog.py --root "$work_context/operators" --output "$stage/operator-index.json"
overlay=''
if [[ -e backend-overlay ]]; then
  [[ -d backend-overlay && ! -L backend-overlay ]] || die "backend-overlay must be a directory"
  if [[ -n "$(find backend-overlay -mindepth 1 ! -type d -print -quit)" ]]; then
    overlay="$work_context/overlay"
    cp -a backend-overlay/. "$overlay/files/"
  fi
fi

mkdir -p "$work_context/docs"
install -m 644 docs/*.md "$work_context/docs/"
install -m 644 "$stage/operator-index.json" tools/operator_catalog.py "$work_context/"
time_label="$(date -u +%Y%m%dT%H%M%SZ)-${stage##*-}"
source_archive="anyinfer-operators-${operator_source_sha:0:12}-$time_label.tar.gz"
tar -C "$stage" -cf - anyinfer-operators | gzip -1 > "$stage/$source_archive"
mv "$stage/$source_archive" "$out/$source_archive"
(cd "$out" && sha256sum "$source_archive" > "$source_archive.sha256")
printf 'Operator sources: %s\n' "$out/$source_archive"
if ((sources_only)); then exit 0; fi

inspect_image() {
  docker image inspect --platform "$platform" --format "$1" "$2" 2>/dev/null ||
    docker image inspect --format "$1" "$2"
}

build_operators() (
  base_tag="anyinfer-build-base:${stage##*-}-$arch-$hw"
  docker tag "$image" "$base_tag"
  trap 'if [[ "$(docker image inspect --format "{{len .RepoTags}}" "$base_tag" 2>/dev/null)" -gt 1 ]]; then docker image rm --no-prune "$base_tag" >/dev/null; fi' EXIT
  ANYINFER_BASE_IMAGE="$base_tag" ANYINFER_IIDFILE="$stage/derived-$arch-$hw.id" \
    bash "$work_context/build-image.sh" "$hw" "$platform"
)

package_platform() (
platform=$1
bundle=$2
arch=${platform#linux/}
builder_override=ANYINFER_BUILDER_IMAGE_${arch^^}
builder=${!builder_override:-$builder}
mkdir -p "$bundle"/{assets/bin,assets/benches,images,build,docs,tools}
install -m 755 "$root/run-reference.sh" "$bundle/run.sh"
install -m 755 "$root/benchmark-reference.sh" "$bundle/benchmark.sh"
install -m 755 "$root/deploy/entrypoint.sh" "$bundle/assets/entrypoint.sh"
install -m 644 "$root/deploy/calibrate.py" "$bundle/assets/calibrate.py"
install -m 644 "$root/benches/cases.json" "$bundle/assets/benches/cases.json"
install -m 644 "$root/benches/accuracy-prompts.jsonl" "$bundle/assets/benches/accuracy-prompts.jsonl"
install -m 644 "$root"/docs/*.md "$bundle/docs/"
install -m 644 "$root/README-reference.md" "$bundle/README.md"
install -m 644 "$root/tools/"{report,operator_report,operator_catalog,communication_report}.py "$bundle/tools/"
install -m 644 "$root/reference/Cargo.lock" "$bundle/build/Cargo.lock"
install -m 644 "$stage/operator-index.json" "$bundle/assets/operator-index.json"
install -m 644 "$work_context/operator-source.sha256" "$bundle/build/operator-source.sha256"
install -m 644 "$work_context/operators/SOURCE.json" "$bundle/build/operator-upstream.json"
cd "$root"
source_inputs=(reference tools benches build-reference.sh benchmark-reference.sh run-reference.sh deploy README-reference.md docs)
if [[ -n "$overlay" ]]; then
  source_inputs+=(backend-overlay)
  (cd "$overlay/files" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$bundle/build/overlay.sha256"
fi
[[ ! -d tests ]] || source_inputs+=(tests)
find "${source_inputs[@]}" -type f ! -path '*/__pycache__/*' ! -path 'reference/target/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$bundle/build/source.sha256"
{
  printf 'built_utc=%s\nplatform=%s\nprofiles=%s\n' "$(date -u +%FT%TZ)" "$platform" "$profiles"
  commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
  printf 'git_commit=%s\n' "$commit"
  printf 'source_manifest_sha256=%s\n' "$(sha256sum "$bundle/build/source.sha256" | cut -d' ' -f1)"
  printf 'operator_source_manifest_sha256=%s\n' "$operator_source_sha"
  printf 'operator_source_archive=%s\n' "$source_archive"
} > "$bundle/build/build.txt"

# A native musl target produces a binary usable across all three CANN base images.
if ((!local_images)); then docker pull --platform "$platform" "$builder"; fi
builder_id=$(inspect_image '{{.Id}}' "$builder")
[[ "$(inspect_image '{{.Os}}/{{.Architecture}}' "$builder")" == "$platform" ]] || die "builder and target architecture differ"
docker run --rm --platform "$platform" -v "$root:/src:ro" -w /src \
  -v "$bundle/assets/bin:/out" -v "$bundle/build:/meta" \
  -v "anyinfer-cargo-registry:/usr/local/cargo/registry" \
  -v "anyinfer-target-$arch:/target" -e CARGO_TARGET_DIR=/target \
  -e "RESULT_OWNER=$(id -u):$(id -g)" "$builder" sh -ec '
    target=$(rustc -vV | sed -n "s/^host: //p")
    rustc -vV > /meta/rustc.txt
    cargo test --manifest-path reference/Cargo.toml --locked --target "$target" --release
    cargo build --manifest-path reference/Cargo.toml --locked --target "$target" --release
    cp "/target/$target/release/anyinfer-reference" /out/anyinfer
    chmod 755 /out/anyinfer
    chown "$RESULT_OWNER" /out/anyinfer /meta/rustc.txt
  '
printf 'builder=%s\nbuilder_id=%s\n' "$builder" "$builder_id" >> "$bundle/build/build.txt"

IFS=, read -r -a selected <<< "$profiles"
save_flags=()
if [[ "$(docker image save --help)" == *--platform* ]] &&
    docker version --format '{{.Server.APIVersion}}' | awk -F. '{exit !($1 > 1 || ($1 == 1 && $2 >= 48))}'; then
  save_flags+=(--platform "$platform")
fi
for hw in "${selected[@]}"; do
  [[ ! -e "$bundle/build/$hw.id" ]] || die "duplicate profile $hw"
  suffix="-$hw"
  [[ "$hw" != a2 ]] || suffix=
  override=ANYINFER_IMAGE_${hw^^}
  image=${!override:-quay.io/ascend/vllm-ascend:v0.30.0$suffix}
  override=ANYINFER_IMAGE_${hw^^}_${arch^^}
  image=${!override:-$image}
  if ((!local_images)); then docker pull --platform "$platform" "$image"; fi
  id=$(inspect_image '{{.Id}}' "$image")
  actual_platform=$(inspect_image '{{.Os}}/{{.Architecture}}' "$image")
  [[ "$actual_platform" == "$platform" ]] || die "$image resolved to $actual_platform"
  docker image inspect --format '{{json .RepoDigests}}' "$image" > "$bundle/build/$hw.repo-digests.json"
  printf '%s\n' "$id" > "$bundle/build/$hw.base.id"
  case "$hw" in
    a2) soc=ascend910b1 ;;
    a3) soc=ascend910_9391 ;;
    a5) soc=ascend950dt_9582 ;;
  esac
  build_operators
  id=$(cat "$stage/derived-$arch-$hw.id")
  printf '%s\n' "$id" > "$bundle/build/$hw.id"
  tagged=anyinfer-runtime:"$arch-$hw-${id#sha256:}"
  docker tag "$id" "$tagged"
  printf '%s\n' "$tagged" > "$bundle/build/$hw.ref"
  inspect_image '{{json .RootFS.Layers}}' "$tagged" > "$bundle/build/$hw.layers.json"
  printf '%s=%s\n' "$hw" "$image" >> "$bundle/build/build.txt"
  docker run --rm --platform "$platform" --entrypoint cat "$tagged" \
    /opt/anyinfer/operators/manifest.json > "$bundle/build/$hw.operators.json"
  python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m["source_manifest_sha256"] == sys.argv[2] and m["device_kernel_files"] > 0 and m["soc"] == sys.argv[3] else 1)' \
    "$bundle/build/$hw.operators.json" "$operator_source_sha" "$soc"
  docker run --rm --platform "$platform" --entrypoint cat "$tagged" \
    /opt/anyinfer/operators/compiler.txt > "$bundle/build/$hw.compiler.txt"
  docker run --platform "$platform" --rm -v "$bundle/assets:/app:ro" \
    --entrypoint /app/bin/anyinfer "$tagged" --help >/dev/null
  docker run --rm --platform "$platform" --entrypoint python3 "$tagged" -c '
import importlib.util, os, pathlib
binary = pathlib.Path(importlib.util.find_spec("vllm").origin).with_name("vllm-rs")
print(f"{binary} executable={os.access(binary, os.X_OK) and binary.is_file()}")
' > "$bundle/build/$hw.frontend.txt"
  if ! grep -q 'executable=True' "$bundle/build/$hw.frontend.txt"; then
    echo "Warning: $image does not contain an executable vllm-rs. Use --frontend python with this image." >&2
  fi
  docker image save "${save_flags[@]}" -o "$bundle/images/$hw.tar" "$tagged"
done
(cd "$bundle" && sha256sum assets/bin/anyinfer assets/entrypoint.sh assets/calibrate.py assets/operator-index.json assets/benches/* run.sh benchmark.sh tools/*.py build/*.id build/*.ref build/*.layers.json build/*.operators.json build/*.frontend.txt build/operator-source.sha256 > build/payload.sha256)
)
bundle="$stage/anyinfer"
if [[ "$platform" == all ]]; then
  mkdir -p "$bundle"/{platforms,docs,tools}
  install -m 755 run-reference.sh "$bundle/run.sh"
  install -m 755 benchmark-reference.sh "$bundle/benchmark.sh"
  install -m 644 docs/package.md "$bundle/docs/package.md"
  install -m 644 README-reference.md "$bundle/README.md"
  for cpu in arm64 amd64; do package_platform "linux/$cpu" "$bundle/platforms/$cpu"; done
else
  package_platform "$platform" "$bundle"
fi
# The unpacked archive can itself be edited and rebuilt on a compiler host.
mv "$work_context/operators" "$bundle/operators"
install -m 755 build-reference.sh benchmark-reference.sh run-reference.sh "$bundle/"
install -m 644 README-reference.md "$bundle/"
install -m 644 Cargo.toml Cargo.lock .gitattributes .gitignore "$bundle/"
source_dirs=(src reference native benches deploy docs tools)
if [[ -d tests ]]; then source_dirs+=(tests); fi
tar --exclude='__pycache__' --exclude='reference/target' -cf - "${source_dirs[@]}" | tar -xf - -C "$bundle"
if [[ -n "$overlay" ]]; then cp -a "$overlay/files" "$bundle/backend-overlay"; fi
archive=anyinfer-"$archive_arch"-"$time_label".tar.gz
printf '%s\n' 'anyinfer-release-v1' > "$bundle/release-format.txt"
tar -C "$stage" -cf - anyinfer | gzip -1 > "$stage/$archive"
(cd "$stage" && sha256sum "$archive" > "$archive.sha256")
python3 deploy/publish_release.py "$stage" "$archive"
printf 'Deployment directory: %s\nBenchmark entry: %s\nPackage: %s\nChecksum: %s\n' \
  "$out/anyinfer" "$out/anyinfer/benchmark.sh" "$out/$archive" "$out/$archive.sha256"
