"""Summarize one isolated Ascend profiler replay. Uses only the Python standard library."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from array import array
from collections import defaultdict
from pathlib import Path


def normalized(header):
    header = header.strip().lower().replace("μ", "u").replace("µ", "u")
    return re.sub("[^a-z0-9]", "", header)


def first(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip().lower() not in {"", "n/a", "na", "nan", "null", "-", "none"}:
            return str(value).strip()
    return None


def number(value):
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite measurement {value}")
    return result


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        keys = {field: normalized(field) for field in fields}
        if not {"durationus", "taskdurationus", "taskduration"}.intersection(keys.values()):
            raise ValueError(f"{path.name} has no supported device duration column")
        # Preserve vendor units in counter labels; report arithmetic means over valid samples.
        pmu = {field: key for field, key in keys.items() if memory_counter(key) or pipe_counter(key)}
        for values in reader:
            if None in values:
                raise ValueError(f"{path.name}:{reader.line_num}: too many columns")
            row = {key: values[field] for field, key in keys.items()}
            counters = {field.strip(): first(row, key) for field, key in pmu.items()}
            yield row, counters


def memory_counter(key):
    return any(word in key for word in ("bandwidth", "readbw", "writebw", "readbytes", "writebytes"))


def pipe_counter(key):
    return (key.startswith(("aic", "aiv", "cube", "vector", "mac", "vec", "mte", "scalar"))
            and any(word in key for word in ("ratio", "cycle", "throughput", "time", "utilization", "fops")))


def pipe_activity(key):
    return (pipe_counter(key) and any(unit in key for unit in ("vec", "mac", "mte", "scalar", "fixp", "cube"))
            and any(field in key for field in ("ratio", "time", "utilization")))


def db_rows(path):
    # CANN COMPUTE_TASK_INFO supplies shape and identity; TASK supplies device timestamps in ns.
    sql = """
        SELECT name.value AS name, kind.value AS type, core.value AS tasktype,
               shape.value AS inputshapes, dtype.value AS inputdatatypes, op.blockDim AS blockdim,
               task.deviceId AS deviceid, task.streamId AS streamid, task.startNs / 1000.0 AS starttimeus,
               (task.endNs - task.startNs) / 1000.0 AS durationus
        FROM TASK AS task JOIN COMPUTE_TASK_INFO AS op USING(globalTaskId)
        LEFT JOIN STRING_IDS AS name ON name.id=op.name
        LEFT JOIN STRING_IDS AS kind ON kind.id=op.opType
        LEFT JOIN STRING_IDS AS core ON core.id=op.taskType
        LEFT JOIN STRING_IDS AS shape ON shape.id=op.inputShapes
        LEFT JOIN STRING_IDS AS dtype ON dtype.id=op.inputDataTypes
    """
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(sql):
            yield dict(row), {}
    finally:
        connection.close()


class Samples:
    def __init__(self):
        self.durations = array("d")
        self.rank_times = defaultdict(float)
        self.counters = {}
        self.names = set()
        self.rank_evidence = {}

    def observe(self, duration, rank, name, counters):
        self.durations.append(duration)
        self.rank_times[rank] += duration
        if len(self.names) < 3:
            self.names.add(name)
        for key, value in counters.items():
            if value is None:
                continue
            # Some exporter versions include a '%' suffix on ratios.
            raw = value.rstrip("%").strip()
            numeric = number(raw)
            label = key + " (%)" if value.endswith("%") and "%" not in key else key
            stats = self.counters.setdefault(label, [0, 0.0, 0.0, 0.0])
            stats[0] += 1
            stats[1] += numeric
            stats[2] += duration
            stats[3] += duration * numeric

    def summarize(self):
        ordered = sorted(self.durations)
        calls = len(ordered)
        total = math.fsum(ordered)
        return {
            "calls": calls, "sum_device_us": total, "mean_us": total / calls,
            "p50_us": ordered[math.ceil(calls * 0.50) - 1],
            "p95_us": ordered[math.ceil(calls * 0.95) - 1], "max_us": ordered[-1],
            "rank_sum_device_us": dict(sorted(self.rank_times.items())),
            "sample_names": sorted(self.names),
            "hardware_counters": {key: {"samples": n, "mean": summed / n}
                                  for key, (n, summed, _, _) in sorted(self.counters.items())},
            "rank_evidence": self.rank_evidence,
        }


def capture_file(directory, group):
    # The text exporter may also retain the CANN CSV and DB behind kernel_details.csv.
    # Choose one representation per rank to avoid counting the same kernel twice.
    fallback = None
    for pattern in ("kernel_details.csv", "op_summary_*.csv", "ascend_pytorch_profiler*.db", "msprof_*.db"):
        files = sorted(directory.rglob(pattern))
        if len(files) > 1:
            if fallback:
                return fallback
            raise ValueError(f"ambiguous rank capture: {len(files)} files match {pattern}")
        if files:
            path = files[0]
            if group not in {"Memory", "PipeUtilization"} or path.suffix == ".db":
                return fallback or path
            with path.open(encoding="utf-8-sig", newline="") as stream:
                headers = [normalized(field) for field in next(csv.reader(stream), [])]
            if {"durationus", "taskdurationus", "taskduration"}.intersection(headers):
                detect = memory_counter if group == "Memory" else pipe_activity
                if any(detect(h) for h in headers):
                    return path
                fallback = fallback or path
    if fallback:
        return fallback
    raise ValueError("no exported kernel detail CSV, op_summary CSV or profiler DB")


def read_rank(rank_dir, rank):
    marker = json.loads((rank_dir / "rank.json").read_text(encoding="utf-8"))
    if marker.get("status") != "exported" or marker.get("rank", marker.get("global_rank")) != rank:
        raise ValueError("missing completed export for the expected rank")
    path = capture_file(rank_dir, marker.get("aic_metrics"))
    events = csv_rows(path) if path.suffix == ".csv" else db_rows(path)
    groups = defaultdict(Samples)
    streams = defaultdict(list)
    calls = 0
    shaped = 0
    for row, counters in events:
        name = first(row, "name", "opname", "kernelname")
        kind = first(row, "type", "optype") or name
        duration = number(first(row, "durationus", "taskdurationus", "taskduration"))
        if not kind or duration is None or duration < 0:
            raise ValueError("unidentified kernel or invalid task duration")
        shape = first(row, "inputshapes", "inputshape")
        key = (kind, first(row, "acceleratorcore", "tasktype"), shape, first(row, "inputdatatypes", "inputtypes"),
               first(row, "inputformats"), first(row, "blockdim", "blocknum"),
               first(row, "outputshapes", "outputshape"), first(row, "outputdatatypes", "outputtypes"),
               first(row, "outputformats"))
        groups[key].observe(duration, rank, name or kind, counters)
        start = number(first(row, "starttimeus", "taskstarttimeus"))
        stream = first(row, "streamid")
        if start is not None and duration > 0:
            streams[(first(row, "deviceid"), stream)].append((start, duration, kind))
        calls += 1
        shaped += shape is not None
    counter_names = sorted({name for samples in groups.values() for name in samples.counters})
    info = {"rank": rank, "worker": marker, "file": str(path.relative_to(rank_dir.parent)), "counter_names": counter_names,
            "calls": calls, "rows_with_input_shape": shaped,
            "scope": "compute tasks" if path.suffix == ".db" else "exported device tasks"}
    return info, groups, streams


def merge_samples(target, group):
    target.durations.extend(group.durations)
    target.rank_times.update(group.rank_times)  # each merged group belongs to a distinct rank
    target.names.update(sorted(group.names)[:max(0, 3 - len(target.names))])
    for key, values in group.counters.items():
        stats = target.counters.setdefault(key, [0, 0.0, 0.0, 0.0])
        for i, value in enumerate(values):
            stats[i] += value


def tensor_shapes(text):
    """Accept only explicit positive dimensions in CANN semicolon or JSON array forms."""
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if not isinstance(parsed, list):
        parts = text.strip('"').split(";")
        parsed = []
        for part in parts:
            dims = part.strip().strip("[]()").split(",")
            if not all(v.strip().isdigit() for v in dims):
                return []
            parsed.append([int(v) for v in dims])
    elif parsed and isinstance(parsed[0], int):
        parsed = [parsed]
    if not all(isinstance(shape, list) and shape
               and all(type(d) is int and 0 < d <= 2**31 for d in shape) for shape in parsed):
        return []
    return parsed


def tokens(text):
    return [v.upper().removeprefix("DT_") for v in re.findall(r"[A-Za-z0-9_]+", text or "")]


def work_model(key):
    kind, core, shapes, types, formats, _, out_shape, out_type, out_format = key
    if normalized(core or "") not in {"aicore", "aicube", "aivectorcore", "aivector"}:
        return None
    ins, outs = tensor_shapes(shapes), tensor_shapes(out_shape)
    dtypes = [{"BFLOAT16": "BF16", "FLOAT32": "FLOAT"}.get(t, t) for t in tokens(types)]
    if (len(outs) != 1 or not ins or len(ins) != len(dtypes) or tokens(out_type) not in (["BF16"], ["BFLOAT16"])
            or tokens(formats) != ["ND"] * len(ins) or tokens(out_format) != ["ND"]):
        return None
    output = outs[0]
    flops = None
    if kind in {"MatMul", "MatMulV2"} and len(ins) == 2 and dtypes == ["BF16"] * 2:
        # Accept transposes only when input/output extents determine a unique FMA count.
        if (normalized(core) not in {"aicore", "aicube"} or len(ins[0]) != 2
                or len(ins[1]) != 2 or len(output) != 2):
            return None
        m, n = output
        ks = {k for a, k in (ins[0], ins[0][::-1]) for b, z in (ins[1], ins[1][::-1])
              if (a, z) == (m, n) and k == b}
        if len(ks) != 1:
            return None
        flops = 2 * m * ks.pop() * n
    elif kind == "HcPost" and len(ins) == 4 and dtypes[:2] == ["BF16", "BF16"]:
        x, residual, post, comb = ins
        if (len(x) not in (2, 3) or not post or post[:-1] != x[:-1] or residual != post + [x[-1]]
                or comb != post + [post[-1]] or output != residual or dtypes[2] not in {"BF16", "FLOAT", "FLOAT16"}
                or dtypes[2] != dtypes[3]):
            return None
    elif (kind in {"Abs", "Neg", "Relu", "Add", "Mul"}
          and len(ins) == (2 if kind in {"Add", "Mul"} else 1)
          and all(s == output for s in ins) and dtypes == ["BF16"] * len(ins)):
        pass
    else:
        return None
    logical_bytes = sum(math.prod(s) * (4 if t == "FLOAT" else 2) for s, t in zip(ins, dtypes))
    model = {"logical_tensor_bytes_per_call": logical_bytes + 2 * math.prod(output)}
    if flops is not None:
        model["dense_flops_per_call"] = flops
        model["logical_flops_per_byte"] = flops / model["logical_tensor_bytes_per_call"]
    return model


def reference_rates(path):
    if not path:
        return {"status": "not_supplied"}, {}
    try:
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if data["schema_version"] != 1:
            raise ValueError("unrecognized calibration schema")
        meta = {"status": data["status"], "sha256": hashlib.sha256(raw).hexdigest()}
        if data["status"] not in {"ok", "partial"}:
            return meta, {}
        rates = {}
        core_counts = {}
        for device in data["devices"]:
            chip = device["physical_device"]
            if type(chip) is not int or chip in rates:
                raise ValueError("ambiguous physical device in calibration")
            best = {}
            rates[chip] = best
            core_counts[chip] = {k: device[k] for k in ("cube_core_num", "vector_core_num")
                                 if type(device.get(k)) is int and device[k] > 0}
            if device.get("status") not in {"ok", "partial"}:
                continue
            for probe in device.get("probes", []):
                if probe.get("reference_eligible") is not True or probe.get("status") != "ok":
                    continue
                for kind, field, dtype in (("bf16_gemm", "measured_tflops", "bfloat16"),
                                           ("stream_copy", "measured_gb_s", "bfloat16"),
                                           ("int8_quant_gemm", "measured_tops", "int8")):
                    rate = probe.get(field)
                    if (probe.get("id") == kind and probe.get("dtype") == dtype
                            and isinstance(rate, (int, float)) and math.isfinite(rate)
                            and rate > best.get(kind, 0)):
                        best[kind] = rate
        meta["rates_by_physical_device"] = rates
        meta["physical_core_counts"] = core_counts
        return meta, rates
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {"status": "unavailable", "error": str(exc)}, {}


def rank_evidence(group, rank, physical, key, refs, cores):
    duration = group.rank_times[rank]
    calls = len(group.durations)
    evidence = {"physical_device": physical, "calls": calls, "mean_task_us": duration / calls,
                "duration_weighted_pmu": {
                    name: {"mean": weighted / weight, "samples": count, "covered_task_us": weight}
                    for name, (count, _, weight, weighted) in group.counters.items() if weight > 0
                    and ("ratio" in normalized(name) or "mainmem" in normalized(name))}}
    core = normalized(key[1] or "")
    model = work_model(key)
    block = key[5]
    domain = None
    if core in {"aivectorcore", "aivector"}:
        domain = "vector_core_num"
    elif core == "aicube" or (core == "aicore" and model and "dense_flops_per_call" in model):
        domain = "cube_core_num"
    if domain in cores and block and block.isdigit() and int(block) > 0:
        evidence["launch_geometry"] = {"blocks": int(block), "device_core_field": domain, "device_cores": cores[domain],
                                       "blocks_per_physical_core": int(block) / cores[domain]}
    if not model or duration <= 0:
        return evidence
    # Use positive-duration invocations only, avoiding attribution of zero-resolution samples.
    timed_calls = sum(d > 0 for d in group.durations)
    traffic = model["logical_tensor_bytes_per_call"]
    bw = traffic * timed_calls / (duration * 1000)
    evidence["logical_gb_s"] = bw
    if "stream_copy" in refs:
        evidence["logical_gb_s_over_copy_reference"] = bw / refs["stream_copy"]
    if "dense_flops_per_call" in model:
        rate = model["dense_flops_per_call"] * timed_calls / (duration * 1e6)
        evidence["dense_tflops"] = rate
        if "bf16_gemm" in refs:
            evidence["dense_tflops_over_gemm_reference"] = rate / refs["bf16_gemm"]
            if "stream_copy" in refs:
                memory_limit = model["logical_flops_per_byte"] * refs["stream_copy"] / 1000
                ridge = min(refs["bf16_gemm"], memory_limit)
                evidence["empirical_roofline"] = {
                    "reference_tflops": ridge, "achieved_over_reference": rate / ridge,
                    "lower_reference_term": "logical_traffic" if memory_limit < refs["bf16_gemm"] else "bf16_compute"}
    return evidence


def task_timeline(streams):
    by_device = defaultdict(list)
    for (device, _), events in streams.items():
        for start, duration, _ in events:
            by_device[device].extend(((start, 1), (start + duration, -1)))
    rows = []
    for device, changes in by_device.items():
        changes.sort()
        start = previous = changes[0][0]
        depth = 0
        busy = overlap = longest_gap = 0.0
        for timestamp, delta in changes:
            dt = timestamp - previous
            if depth > 0:
                busy += dt
            elif dt > longest_gap:
                longest_gap = dt
            if depth > 1:
                overlap += dt
            depth += delta
            previous = timestamp
        span = previous - start
        rows.append({"export_device_id": device, "timed_calls": len(changes) // 2,
                     "first_to_last_task_us": span, "task_union_us": busy, "concurrent_task_us": overlap,
                     "no_listed_task_us": span - busy, "longest_no_listed_task_us": longest_gap,
                     "task_coverage_fraction": busy / span if span else None})
    return rows


def observed_main_memory_rates(local, rank):
    # Compare the same PMU export, core type, rank and replay. Keep the reference
    # workload visible; the largest measured group can still be far below capacity.
    comparable = defaultdict(list)
    for key, group in local.items():
        for label, (count, _, weight, weighted) in group.counters.items():
            if (re.fullmatch(r"(aic|aiv)?mainmem(read|write)bwgbs", normalized(label))
                    and count >= 3 and weight > 0 and weighted >= 0):
                comparable[(key[1], label)].append((key, weighted / weight, count))
    references = []
    for (core, label), rows in comparable.items():
        if len(rows) < 2:
            continue
        best, rate, count = max(rows, key=lambda row: row[1])
        if rate == 0:
            continue
        references.append({"task_type": core, "counter": label, "highest_observed_group_mean": rate,
                           "op_type": best[0], "input_shapes": best[2], "input_dtypes": best[3],
                           "block_dim": best[5], "samples": count, "eligible_groups": len(rows)})
        for key, value, _ in rows:
            local[key].rank_evidence[rank]["duration_weighted_pmu"][label]["over_highest_observed_group"] = value / rate
    return references


def summarize(trace, expected, index, calibration=None):
    if expected < 1:
        raise ValueError("expected rank count must be positive")
    report = {"schema_version": 1, "expected_ranks": expected, "errors": [], "warnings": [], "ranks": [], "operators": []}
    report["calibration"], reference = reference_rates(calibration)
    symbols = {}
    if index:
        data = index.read_bytes()
        catalog = json.loads(data)
        symbols = catalog["symbols"]
        report.update({"catalog_sha256": hashlib.sha256(data).hexdigest(),
                       "source_tree_sha256": catalog["tree_sha256"]})
    groups = defaultdict(Samples)
    pairs = defaultdict(lambda: [0, 0.0])
    expected_dirs = {f"rank-{r}" for r in range(expected)}
    extra = sorted(p.name for p in trace.glob("rank-*") if p.name not in expected_dirs)
    if extra:
        report["errors"].append(f"unexpected rank directories: {extra}")
    for rank in range(expected):
        try:
            info, local, streams = read_rank(trace / f"rank-{rank}", rank)
            info["timelines"] = task_timeline(streams)
            report["ranks"].append(info)
            if info["calls"] and not info["rows_with_input_shape"]:
                report["warnings"].append(f"rank {rank}: device export contains no input shapes")
            if info["calls"] and not info["counter_names"]:
                report["warnings"].append(f"rank {rank}: no numeric hardware counters exported")
            if (info["calls"] and info["worker"].get("aic_metrics") == "PipeUtilization"
                    and not any(pipe_activity(normalized(name)) for name in info["counter_names"])):
                report["warnings"].append(f"rank {rank}: no numeric pipeline activity counters exported")
            if (info["calls"] and info["worker"].get("aic_metrics") == "Memory"
                    and not any(memory_counter(normalized(name)) for name in info["counter_names"])):
                report["errors"].append(f"rank {rank}: requested Memory PMU group produced no memory counters")
            for key, group in local.items():
                physical = info["worker"].get("physical_device")
                cores = report["calibration"].get("physical_core_counts", {}).get(physical, {})
                group.rank_evidence[rank] = rank_evidence(group, rank, physical, key, reference.get(physical, {}), cores)
            info["observed_main_memory_rates"] = observed_main_memory_rates(local, rank)
            for key, group in local.items():
                merge_samples(groups[key], group)
                groups[key].rank_evidence.update(group.rank_evidence)
            for (_, stream), events in streams.items():
                if stream is None:
                    continue
                events.sort(key=lambda ev: ev[0])
                for (start, duration, previous), (nxt, _, following) in zip(events, events[1:]):
                    gap = nxt - start - duration
                    if gap >= 0:
                        stat = pairs[(previous, following)]
                        stat[0] += 1
                        stat[1] += gap
        except (ValueError, OSError, KeyError, sqlite3.Error, csv.Error) as exc:
            report["errors"].append(f"rank {rank}: {exc}")
    for key, stats in groups.items():
        kind, core, shape, dtype, layout, block, outshape, outdtype, outlayout = key
        row = stats.summarize()
        row.update({"op_type": kind, "task_type": core, "input_shapes": shape, "input_dtypes": dtype,
                    "input_formats": layout, "block_dim": block, "output_shapes": outshape,
                    "output_dtypes": outdtype, "output_formats": outlayout, "work_model": work_model(key)})
        matched = {name: symbols[name] for name in (kind, *row["sample_names"]) if name in symbols}
        row["source_candidates_by_exact_symbol"] = matched
        report["operators"].append(row)
    report["operators"].sort(key=lambda row: -row["sum_device_us"])
    report["sum_device_us"] = math.fsum(o["sum_device_us"] for o in report["operators"])
    comparable = sum(s.rank_times[rank] for s in groups.values() for rank, ev in s.rank_evidence.items()
                     if "logical_gb_s_over_copy_reference" in ev or "dense_tflops_over_gemm_reference" in ev)
    report["reference_comparison_coverage"] = {
        "comparable_task_us": comparable, "total_task_us": report["sum_device_us"],
        "fraction_of_summed_task_time": comparable / report["sum_device_us"] if report["sum_device_us"] else None}
    report["rows_with_input_shape"] = sum(r["rows_with_input_shape"] for r in report["ranks"])
    if report["sum_device_us"] <= 0:
        report["errors"].append("no positive device duration in this replay")
    report["same_stream_neighbors"] = [
        {"first": pair[0], "next": pair[1], "observations": n, "sum_gap_us": gap}
        for pair, (n, gap) in sorted(pairs.items(), key=lambda p: -p[1][0])[:100]
    ]
    report["ok"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=os.environ.get("ANYINFER_OPERATOR_INDEX"))
    parser.add_argument("--calibration", type=Path, default=os.environ.get("ANYINFER_CALIBRATION"))
    args = parser.parse_args()
    try:
        report = summarize(args.trace_dir, args.expected_ranks, args.index, args.calibration)
    except (KeyError, OSError, ValueError) as exc:
        report = {"schema_version": 1, "ok": False, "errors": [str(exc)]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["ok"]:
        print("; ".join(report["errors"]), file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
