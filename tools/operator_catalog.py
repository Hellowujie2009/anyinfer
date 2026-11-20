"""Index exact backend operator registrations and Triton JIT symbols in the source release."""

import argparse
import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def catalog(root):
    upstream = root / "vllm-ascend"
    csrc = upstream / "csrc"
    symbols = defaultdict(list)

    def rel(path):
        return "operators/" + path.relative_to(root).as_posix()

    def record(symbol, file, kind, sources):
        symbols[symbol].append({"kind": kind, "definition": rel(file),
                                "source_paths": [rel(p) for p in sources]})

    for file in sorted(csrc.glob("*/*/op_host/**/*.cpp")):
        source = file.read_text(encoding="utf-8", errors="replace")
        names = re.findall(r"^\s*OP_ADD\s*\(\s*([A-Za-z_]\w*)\s*[,)]", source, re.M)
        if not names:
            continue
        op = next(p.parent for p in file.parents if p.name == "op_host")
        sources = [p for p in (op / "op_kernel", op / "op_kernel_aicpu", op / "op_host")
                   if p.is_dir()]
        for name in sorted(set(names)):
            record(name, file, "CANN registration", sources)

    for file in sorted((upstream / "vllm_ascend").rglob("*.py")):
        source = file.read_text(encoding="utf-8")
        if "triton.jit" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if (isinstance(target, ast.Attribute) and target.attr == "jit"
                        and isinstance(target.value, ast.Name) and target.value.id == "triton"):
                    record(node.name, file, "Triton JIT", [file])
                    break
    if not symbols:
        raise ValueError(f"no operator registrations in {upstream}")
    # Hash source contents as well as symbols: a tile-size-only change must change provenance.
    digest = hashlib.sha256()
    for file in sorted(p for p in upstream.rglob("*") if p.is_file()):
        name = file.relative_to(upstream)
        if "__pycache__" in name.parts or ".git" in name.parts:
            continue
        digest.update(name.as_posix().encode() + b"\0" + hashlib.sha256(file.read_bytes()).digest())
    return {"schema_version": 1, "tree_sha256": digest.hexdigest(), "symbols": dict(sorted(symbols.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = catalog(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
