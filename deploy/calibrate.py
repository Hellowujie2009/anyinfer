"""Measure per-device GEMM and same-device copy rates before loading the model."""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path


WARMUP = 5
REPEATS = 7
OPS_PER_SAMPLE = 8
COPY_MIB = (32, 256)
GEMM_SIZES = (2048, 4096)


def event_samples_ms(npu, operation, warmup=WARMUP, repeats=REPEATS, ops_per_sample=OPS_PER_SAMPLE):
    """Return per-call milliseconds measured by events on the current device stream."""
    if min(repeats, ops_per_sample) < 1 or warmup < 0:
        raise ValueError("invalid timing window")
    for _ in range(warmup):
        operation()
    npu.synchronize()
    windows = [(npu.Event(enable_timing=True), npu.Event(enable_timing=True)) for _ in range(repeats)]
    for start, end in windows:
        start.record()
        for _ in range(ops_per_sample):
            operation()
        end.record()
    windows[-1][1].synchronize()
    samples = [start.elapsed_time(end) / ops_per_sample for start, end in windows]
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("invalid NPU event duration")
    return samples


def probe_definition(kind, size):
    if kind == "bf16_gemm":
        return {"id": kind, "shape": [size, size, size], "dtype": "bfloat16",
                "operation": "torch.mm(A[M,K], B[K,N], out=C[M,N])",
                "useful_flops_per_call": 2 * size ** 3, "reference_eligible": True}
    if kind == "int8_quant_gemm":
        return {"id": kind, "shape": [size, size, size], "dtype": "int8", "output_dtype": "bfloat16",
                "operation": "npu_quant_matmul, prequantized INT8 operands, no bias",
                "weight_format": "FRACTAL_NZ", "scale": "per token fp32 and per column bf16",
                "useful_integer_ops_per_call": 2 * size ** 3, "reference_eligible": True}
    if kind == "stream_copy":
        buffer_bytes = size * 1024 ** 2
        return {"id": kind, "shape": [buffer_bytes // 2], "dtype": "bfloat16",
                "operation": "dst.copy_(src) on one NPU, disjoint contiguous buffers",
                "bytes_per_buffer": buffer_bytes, "logical_bytes_per_call": 2 * buffer_bytes,
                "reference_eligible": size == max(COPY_MIB)}
    raise ValueError(f"unknown probe: {kind}")


def summarize_samples(probe, samples):
    median = statistics.median(samples)
    result = {"samples_ms": samples, "median_ms": median}
    if probe["id"] == "bf16_gemm":
        result["measured_tflops"] = probe["useful_flops_per_call"] / (median * 1e9)
    elif probe["id"] == "int8_quant_gemm":
        result["measured_tops"] = probe["useful_integer_ops_per_call"] / (median * 1e9)
    else:
        result["measured_gb_s"] = probe["logical_bytes_per_call"] / (median * 1e6)
    return result


def run_probe(torch, device, probe):
    if probe["id"] in {"bf16_gemm", "int8_quant_gemm"}:
        m, k, n = probe["shape"]
        rng = torch.Generator(device="cpu").manual_seed(3121)
        # Small integer BF16 inputs permit a sampled CPU FP32 verification with
        # exact partial sums. Input generation and layout setup are outside timing.
        dtype = torch.bfloat16 if probe["id"] == "bf16_gemm" else torch.int8
        a_host = torch.randint(-1, 2, (m, k), dtype=torch.int8, generator=rng)
        b_host = torch.randint(-1, 2, (k, n), dtype=torch.int8, generator=rng)
        a = a_host.to(device, dtype)
        b = b_host.to(device, dtype)
        row_ids = [0, 1, m // 3, m // 2, m - 1]
        col_ids = [0, 1, n // 3, n // 2, n - 1]
        rows = torch.tensor(row_ids, device=device)
        cols = torch.tensor(col_ids, device=device)
        reference = a_host[row_ids].float() @ b_host[:, col_ids].float()
        del a_host, b_host
        if dtype == torch.bfloat16:
            out = torch.full((m, n), float("nan"), device=device, dtype=dtype)
            def operation():
                return torch.mm(a, b, out=out)
        else:
            import torch_npu

            # Match the vendored W8A8_DYNAMIC Linear branch at default weight_nz_mode=1.
            # Logical (K,N) weights are packed once; input quantization is upstream of this probe.
            b = torch_npu.npu_format_cast(b, 29)
            if int(torch_npu.get_npu_format(b)) != 29:
                raise RuntimeError("W8A8 reference requires FRACTAL_NZ packed weights")
            x_scale = torch.full((m,), 0.5, device=device, dtype=torch.float32)
            w_scale = torch.full((n,), 0.25, device=device, dtype=torch.bfloat16)
            reference *= 0.125
            out = None

            def operation():
                nonlocal out
                out = torch_npu.npu_quant_matmul(a, b, w_scale, pertoken_scale=x_scale,
                                                 bias=None, output_dtype=torch.bfloat16)
        samples = event_samples_ms(torch.npu, operation)
        actual = out.index_select(0, rows).index_select(1, cols).cpu()
        torch.testing.assert_close(actual, reference.bfloat16(), atol=0, rtol=0)
    else:
        count, = probe["shape"]
        src = torch.empty(count, dtype=torch.bfloat16, device=device).uniform_(-1, 1)
        out = torch.empty_like(src)
        out.fill_(float("nan"))
        # Include both ends and spread verification across the full allocation.
        positions = torch.tensor([i * (count - 1) // 256 for i in range(257)], device=device)
        reference = src.index_select(0, positions).cpu()
        samples = event_samples_ms(torch.npu, lambda: out.copy_(src, non_blocking=True))
        actual = out.index_select(0, positions).cpu()
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    return {**summarize_samples(probe, samples), "verified_elements": actual.numel(), "status": "ok"}


def save(output, report):
    # Readers can open the last finished probe even if the process is interrupted.
    tmp = output.with_name(output.name + ".part")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(output)


def parse_devices(raw):
    values = raw.split(",")
    if not all(value.isascii() and value.isdecimal() for value in values):
        raise ValueError(f"expected comma-separated device IDs, got {raw!r}")
    devices = [int(value) for value in values]
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate device IDs in {raw!r}")
    return devices


def calibrate(output, selected, skip=False):
    report = {"schema_version": 1, "status": "running", "devices": [],
              "hardware_family": os.environ.get("ANYINFER_HARDWARE"),
              "requested_devices": selected, "start_unix_ns": time.time_ns(),
              "warmup_calls": WARMUP, "samples": REPEATS, "calls_per_sample": OPS_PER_SAMPLE,
              "clock": "current-stream NPU event elapsed milliseconds",
              "scope": "Sequential single-device measurements before model loading",
              "bandwidth_basis": "Logical source read plus destination write. Cache residency and copy engine can affect results."}
    if skip:
        report.update(status="skipped", end_unix_ns=time.time_ns())
        save(output, report)
        return report
    save(output, report)
    try:
        device_ids = parse_devices(selected)
        if parse_devices(os.environ["ASCEND_RT_VISIBLE_DEVICES"]) != device_ids:
            raise RuntimeError("torch visibility order differs from requested devices")
        import torch
        import torch_npu

        report.update(torch_version=str(torch.__version__), torch_npu_version=str(torch_npu.__version__))
        try:
            report["cann_version"] = str(torch.npu.get_cann_version())
        except (AttributeError, RuntimeError) as exc:
            report["cann_version_error"] = str(exc)
        if torch.npu.device_count() != len(device_ids):
            raise RuntimeError("visible NPU count differs from requested devices")
        if report["hardware_family"] in {"a2", "a3"}:
            # The vendored V1 model runner enables internal layout support before
            # W8A8 weights are packed. This setting is local to the short-lived probe.
            torch.npu.config.allow_internal_format = True
        with torch.inference_mode():
            for ordinal, physical in enumerate(device_ids):
                item = {"ordinal": ordinal, "physical_device": physical, "probes": [], "status": "running",
                        "start_unix_ns": time.time_ns()}
                report["devices"].append(item)
                try:
                    torch.npu.set_device(ordinal)
                    item["device_name"] = torch.npu.get_device_name(ordinal)
                    try:
                        props = torch.npu.get_device_properties(ordinal)
                        for field in ("total_memory", "cube_core_num", "vector_core_num"):
                            reported = getattr(props, field, None)
                            if isinstance(reported, int) and reported > 0:
                                item[field] = reported
                    except (AttributeError, RuntimeError) as exc:
                        item["device_properties_error"] = str(exc)
                    torch.npu.manual_seed(3121)
                    kinds = [("bf16_gemm", GEMM_SIZES), ("stream_copy", COPY_MIB)]
                    if report["hardware_family"] in {"a2", "a3"}:
                        kinds.append(("int8_quant_gemm", GEMM_SIZES))
                    for kind, sizes in kinds:
                        for size in sizes:
                            probe = probe_definition(kind, size)
                            try:
                                probe.update(run_probe(torch, f"npu:{ordinal}", probe))
                            except Exception as exc:
                                probe.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                            item["probes"].append(probe)
                            save(output, report)
                    item["status"] = probe_status(item["probes"])
                    torch.npu.empty_cache()
                except Exception as exc:
                    item.update(status=probe_status(item["probes"]), error=f"{type(exc).__name__}: {exc}")
                item["end_unix_ns"] = time.time_ns()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["status"] = probe_status(report["devices"])
    if "error" in report and report["status"] == "ok":
        report["status"] = "partial"
    report["end_unix_ns"] = time.time_ns()
    save(output, report)
    return report


def probe_status(entries):
    statuses = [entry["status"] for entry in entries]
    if statuses and all(status == "ok" for status in statuses):
        return "ok"
    if any(status in {"ok", "partial"} for status in statuses):
        return "partial"
    return "unavailable"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    result = calibrate(args.output, args.devices, args.skip)
    print(f"Hardware calibration: {result['status']} ({args.output})", file=sys.stderr)
