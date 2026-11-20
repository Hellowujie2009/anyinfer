"""Install the supplied backend edits into the selected image's actual package trees."""

import importlib.util
import shutil
import sys
from pathlib import Path


def install(overlay):
    targets = {}
    for package in sorted(overlay.iterdir()):
        if package.name not in {"vllm", "vllm_ascend"} or package.is_symlink() or not package.is_dir():
            raise ValueError(f"expected vllm/ or vllm_ascend/ under backend-overlay, got {package.name}")
        spec = importlib.util.find_spec(package.name)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f"{package.name} is missing in this image")
        root = Path(next(iter(spec.submodule_search_locations))).resolve(strict=True)
        for src in sorted(package.rglob("*")):
            if src.is_symlink():
                raise ValueError(f"symlink in backend-overlay: {src}")
            if src.is_dir():
                continue
            if not src.is_file():
                raise ValueError(f"expected a regular file: {src}")
            if package.name == "vllm_ascend" and (
                "_cann_ops_custom" in src.parts or ".so" in src.suffixes
            ):
                raise ValueError(f"edit the operator sources under operators/vllm-ascend to rebuild {src.name}")
            dst = (root / src.relative_to(package)).resolve()
            if root not in dst.parents:
                raise ValueError(f"overlay destination escapes {root}: {dst}")
            if src.suffix == ".py":
                compile(src.read_bytes(), str(dst), "exec")
            targets[src] = dst
    for src, dst in targets.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if dst.suffix == ".py":
            for cached in (dst.parent / "__pycache__").glob(dst.stem + ".*.pyc"):
                cached.unlink()


if __name__ == "__main__":
    install(Path(sys.argv[1]))
