"""Run the built HcPost on the first visible NPU and compare every element."""

import ctypes
import json
import os
import statistics
import sys
from pathlib import Path


def check_op_api(library, vendor):
    """Resolve the symbols in the already loaded vendor library, including their owners."""
    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("filename", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("name", ctypes.c_char_p),
            ("address", ctypes.c_void_p),
        ]

    dladdr = ctypes.CDLL("libdl.so.2").dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    handle = ctypes.CDLL(str(library), mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    symbols = {}
    for name in ("aclnnHcPostGetWorkspaceSize", "aclnnHcPost"):
        owner = DlInfo()
        if not dladdr(ctypes.cast(getattr(handle, name), ctypes.c_void_p), ctypes.byref(owner)):
            raise RuntimeError(f"unable to identify library defining {name}")
        path = Path(os.fsdecode(owner.filename)).resolve(strict=True)
        if not path.is_relative_to(vendor):
            raise RuntimeError(f"unexpected provider for {name}: {path}")
        symbols[name] = str(path)
    return symbols


def inputs_and_reference(torch, tokens, hidden, streams):
    rng = torch.Generator(device="cpu").manual_seed(20261120)
    def integers(shape):
        return torch.randint(-4, 5, shape, generator=rng)
    x = integers((1, tokens, hidden)).bfloat16()
    residual = integers((1, tokens, streams, hidden)).bfloat16()
    post = integers((1, tokens, streams)).float() / 8
    comb = integers((1, tokens, streams, streams)).float() / 8
    expected = post.unsqueeze(-1) * x.float().unsqueeze(-2)
    expected += torch.einsum("bsij,bsid->bsjd", comb, residual.float())
    return [value.to("npu:0") for value in (x, residual, post, comb)], expected


def measure_shapes(torch, config_path):
    from calibrate import event_samples_ms

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = config.get("text_config") or config
    hidden, streams = config.get("hidden_size"), config.get("hc_mult")
    if not all(isinstance(value, int) and value > 0 for value in (hidden, streams)):
        return {"status": "unavailable", "reason": "checkpoint does not specify hidden_size and hc_mult"}
    rows = []
    for tokens in (1, 8, 32, 128, 512):
        inputs, expected = inputs_and_reference(torch, tokens, hidden, streams)
        def launch():
            return torch.ops._C_ascend.npu_hc_post(*inputs)
        actual = launch()
        try:
            torch.testing.assert_close(actual.cpu(), expected.bfloat16(), rtol=0, atol=0)
        except AssertionError as exc:
            raise ValueError(f"HcPost numerical mismatch at tokens={tokens}, hidden={hidden}, streams={streams}: {exc}") from exc
        useful_bytes = sum(t.numel() * t.element_size() for t in (*inputs, actual))
        row = {"tokens": tokens, "hidden_size": hidden, "streams": streams, "useful_input_output_bytes": useful_bytes,
               "input_dtypes": [str(t.dtype) for t in inputs], "validated_elements": actual.numel()}
        try:
            samples = event_samples_ms(torch.npu, launch, warmup=5, repeats=7, ops_per_sample=8)
            median_ms = statistics.median(samples)
            row.update(status="ok", ms_per_call=samples, median_us=median_ms * 1000,
                       effective_gb_s=useful_bytes / (1e6 * median_ms))
        except (RuntimeError, ValueError) as exc:
            row.update(status="timing_unavailable", error=str(exc))
        rows.append(row)
    measured = sum(row["status"] == "ok" for row in rows)
    status = "ok" if measured == len(rows) else ("partial" if measured else "unavailable")
    return {"status": status, "physical_device": int(os.environ["ANYINFER_PROBE_DEVICE"]),
            "scope": "isolated warm repeated calls on the first selected NPU; output allocation is part of the op",
            "bandwidth_basis": "logical input and output tensor bytes per call; cache reuse is possible",
            "warmup": 5, "calls_per_sample": 8, "samples_per_shape": 7, "shapes": rows}


def run():
    import torch
    import torch_npu  # noqa: F401 - registers the NPU device

    package_root = Path("/opt/anyinfer/operators/python").resolve(strict=True)
    vendor = (package_root / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer").resolve(strict=True)
    if Path(os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":")[0]).resolve() != vendor:
        raise RuntimeError("the rebuilt vendor must head ASCEND_CUSTOM_OPP_PATH before importing the extension")
    from vllm_ascend import vllm_ascend_C

    extension = Path(vllm_ascend_C.__file__).resolve(strict=True)
    if not extension.is_relative_to(package_root):
        raise RuntimeError(f"unexpected PyTorch extension: {extension}")
    torch.npu.set_device(0)
    # Small binary fractions yield exactly representable sums. Random non-symmetric
    # mixing detects input swaps, matrix transposition and missing residual terms.
    inputs, expected = inputs_and_reference(torch, 3, 4224, 4)
    actual = torch.ops._C_ascend.npu_hc_post(*inputs)
    torch.npu.synchronize()
    result = actual.cpu()
    torch.testing.assert_close(result, expected.bfloat16(), rtol=0, atol=0)
    symbols = check_op_api(vendor / "op_api/lib/libcust_opapi.so", vendor)
    report = {
        "status": "ok",
        "op": "_C_ascend::npu_hc_post",
        "device": torch.npu.get_device_name(0),
        "shape": list(result.shape),
        "dtype": str(result.dtype),
        "compared_elements": result.numel(),
        "max_abs_error": (result.float() - expected).abs().max().item(),
        "extension": str(extension),
        "entry_points": symbols,
    }
    config = os.environ.get("ANYINFER_HC_BENCH_CONFIG")
    if config:
        report["microbenchmark"] = measure_shapes(torch, config)
    return report


if __name__ == "__main__":
    # Keep CANN and framework logs on stderr even if native libraries log via fd 1.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w") as report:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        try:
            print(json.dumps(run()), file=report)
        except Exception as exc:
            print(json.dumps({"status": "failed", "op": "_C_ascend::npu_hc_post", "error": str(exc)}), file=report)
            raise
