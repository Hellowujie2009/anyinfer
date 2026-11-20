"""Summarize one or more extracted benchmark runs, including interrupted runs."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

if __package__:
    from .communication_report import communication_summary
    from .operator_report import reference_rates
else:
    from communication_report import communication_summary
    from operator_report import reference_rates


SAMPLE = re.compile(r'^(vllm:[\w:]+)(\{(?:[^"{}]|"(?:\\.|[^"\\])*")*\})?\s+(\S+)')


def read_samples(path):
    if not path.exists():
        return {}
    samples = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            match = SAMPLE.match(line)
            if match:
                value = float(match[3])
                if math.isfinite(value):
                    samples.setdefault(match[1], {})[match[2]] = value
    return samples


def counter_delta(before, after, key):
    starts = before.get(key, {})
    ends = after.get(key, {})
    if not ends or starts.keys() != ends.keys():
        return None
    changes = [value - starts[label] for label, value in ends.items()]
    return sum(changes) if min(changes) >= 0 else None


def server_stats(directory):
    before = read_samples(directory / "before.prom")
    after = read_samples(directory / "after.prom")
    stats = {}
    for phase in ("queue", "prefill", "decode"):
        prefix = f"vllm:request_{phase}_time_seconds"
        total = counter_delta(before, after, prefix + "_sum")
        count = counter_delta(before, after, prefix + "_count")
        stats[phase + "_mean_ms"] = total * 1000 / count if total is not None and count else None
        stats[phase + "_observations"] = count
    stats["preemptions"] = counter_delta(before, after, "vllm:num_preemptions_total")
    hits = counter_delta(before, after, "vllm:prefix_cache_hits_total")
    queries = counter_delta(before, after, "vllm:prefix_cache_queries_total")
    stats["prefix_hit_fraction"] = hits / queries if hits is not None and queries else None
    gauge_file = directory / "engine-gauges.jsonl"
    if gauge_file.is_file():
        peaks = {}
        attempts = errors = 0
        with gauge_file.open(encoding="utf-8") as source:
            for line in source:
                point = json.loads(line)
                attempts += 1
                errors += int("error" in point)
                for sample in point.get("metrics", []):
                    found = SAMPLE.match(sample)
                    if not found or found[1] not in {"vllm:num_requests_running", "vllm:num_requests_waiting",
                                                     "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"}:
                        continue
                    name = found[1] + (found[2] or "")
                    value = float(found[3])
                    if math.isfinite(value):
                        peaks[name] = max(peaks.get(name, value), value)
        stats["gauge_polls"] = {"attempts": attempts, "errors": errors, "peak_by_series": peaks}
    return stats


def operator_numbers(path):
    data = json.loads((path / "operators.json").read_text(encoding="utf-8"))
    by_type = {}
    for row in data["operators"]:
        stats = by_type.setdefault(row["op_type"], {"calls": 0, "sum_device_us": 0.0})
        stats["calls"] += row["calls"]
        stats["sum_device_us"] += row["sum_device_us"]
    return data, by_type


def hotspot_summary(directory):
    data, types = operator_numbers(directory)
    ranks = data.get("ranks", [])
    shapes = []
    for row in sorted(data["operators"], key=lambda op: -op["sum_device_us"])[:6]:
        rank_times = row.get("rank_sum_device_us", {})
        heaviest = max(rank_times, key=rank_times.get, default=None)
        shapes.append({key: row.get(key) for key in ("op_type", "task_type", "input_shapes", "input_dtypes",
                                                    "block_dim", "calls", "sum_device_us", "work_model")})
        shapes[-1].update({"most_device_time_rank": heaviest,
                           "that_rank_evidence": row.get("rank_evidence", {}).get(heaviest)})
    return {"ok": data["ok"], "errors": data.get("errors", []), "warnings": data.get("warnings", []),
            "path": str(directory / "operators.json"), "source_tree_sha256": data.get("source_tree_sha256"),
            "calibration": data.get("calibration"), "rows_with_input_shape": data.get("rows_with_input_shape"),
            "reference_comparison_coverage": data.get("reference_comparison_coverage"),
            "expected_ranks": data.get("expected_ranks"), "exported_ranks": [r["rank"] for r in ranks],
            "rank_task_timelines": {r["rank"]: r.get("timelines", []) for r in ranks},
            "observed_main_memory_rates": {r["rank"]: r.get("observed_main_memory_rates", []) for r in ranks},
            "hardware_counter_columns": sorted({name for r in ranks for name in r.get("counter_names", [])}),
            "top_by_sum_device_us": [{"op_type": kind, **stats} for kind, stats in
                                     sorted(types.items(), key=lambda p: -p[1]["sum_device_us"])[:12]],
            "top_shapes_by_sum_device_us": shapes}


def golden_status(directory, digest):
    if not digest:
        return "unchecked"
    try:
        with (directory / "accuracy.jsonl").open(encoding="utf-8") as source:
            checks = [json.loads(line) for line in source if line.strip()]
        with (directory / "golden.jsonl").open(encoding="utf-8") as source:
            expected = [json.loads(line)["id"] for line in source if line.strip()]
        if any(not row["ok"] for row in checks):
            return "failed"
        if expected and sorted(row["id"] for row in checks) == sorted(expected):
            return "passed"
    except (KeyError, ValueError, FileNotFoundError):
        pass
    return "incomplete"


def case_environment(directory, capture):
    # Packaged cases are under <host run>/bench/run-*/case-*. A copied case may
    # only have the rank layout and its inner manifest.
    context = {}
    ranks = capture.get("ranks", [])
    if len(ranks) == capture["expected_ranks"] and all(
            isinstance(rank.get("worker", {}).get(k), int) for rank in ranks for k in ("tp_rank", "dp_rank")):
        context["rank_layout"] = sorted([r["rank"], r["worker"]["tp_rank"], r["worker"]["dp_rank"]] for r in ranks)
    if directory.parent.parent.name != "bench":
        return context
    root = directory.parent.parent.parent
    selection = root / "selection.txt"
    if selection.is_file():
        family = re.search(r"^hardware=(a[235])$", selection.read_text(encoding="utf-8"), flags=re.MULTILINE)
        if family:
            context["hardware_family"] = family[1]
    for key, filename, field in (("hardware_family", "status.json", "hardware"),
                                 ("device_model", "operator-smoke.json", "device")):
        path = root / filename
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8")).get(field)
            if value:
                context[key] = value
    for key, filename in (("checkpoint_metadata_sha256", "checkpoint-metadata.sha256"),
                          ("supplied_weight_manifest_sha256", "checkpoint-supplied-weights.sha256"),
                          ("build_inputs_sha256", "build/source.sha256")):
        path = root / filename
        if path.is_file():
            context[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return context


def compare_cases(old_dir, new_dir):
    captures = []
    for directory in (old_dir, new_dir):
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory.parent / "manifest.json").read_text(encoding="utf-8"))
        data, types = operator_numbers(directory)
        if not data["ok"] or not summary["ok"]:
            raise ValueError(f"incomplete benchmark or operator replay: {directory}")
        primary = summary["operator_profile"]
        workload = (summary["case"], primary["replay"], data["expected_ranks"],
                    manifest["prompt_generator"], manifest.get("text_prompt_generator"),
                    primary.get("aic_metrics"), primary.get("request_phase", "operator_replay"))
        golden = manifest.get("golden_sha256")
        captures.append((workload, summary, data, types, golden, golden_status(directory.parent, golden)))
    old, new = captures
    if old[0] != new[0]:
        raise ValueError("different case definitions, profiling replays, prompt generators, PMU groups or rank counts")
    rows = []
    for kind in sorted(old[3].keys() | new[3].keys()):
        before = old[3].get(kind, {"sum_device_us": 0, "calls": 0})
        after = new[3].get(kind, {"sum_device_us": 0, "calls": 0})
        rows.append({"op_type": kind, "baseline": before, "candidate": after,
                     "delta_sum_device_us": after["sum_device_us"] - before["sum_device_us"]})
    rows.sort(key=lambda row: -abs(row["delta_sum_device_us"]))
    base_env, new_env = case_environment(old_dir, old[2]), case_environment(new_dir, new[2])
    identity = ("hardware_family", "device_model", "checkpoint_metadata_sha256", "supplied_weight_manifest_sha256",
                "rank_layout")
    mismatches = [k for k in identity if k in base_env and k in new_env and base_env[k] != new_env[k]]
    if mismatches:
        raise ValueError("different hardware, checkpoint or rank topology: " + ", ".join(mismatches))
    return {"case_id": old[1]["case"]["id"], "baseline_dir": str(old_dir), "candidate_dir": str(new_dir),
            "same_golden_passed": old[5] == new[5] == "passed" and old[4] == new[4],
            "baseline_environment": base_env, "candidate_environment": new_env,
            "unverified_environment_fields": [k for k in identity if k not in base_env or k not in new_env],
            "source_trees": [c[2].get("source_tree_sha256") for c in captures],
            "compared_primary_pmu_group": old[0][-2],
            "compared_primary_request_phase": old[0][-1],
            "requested_pmu_groups": [c[1]["operator_profile"].get("requested_groups",
                                       [c[1]["operator_profile"].get("aic_metrics")]) for c in captures],
            "calibrations": [c[2].get("calibration") for c in captures],
            "baseline_measurement": old[1]["measurement"], "candidate_measurement": new[1]["measurement"],
            "operator_replay_deltas": rows}


def summarize(manifest_path):
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate_path = directory / "summary.json"
    completed = aggregate_path.is_file()
    report = {
        "directory": str(directory),
        "suite_sha256": manifest["suite_sha256"],
        "completed": completed,
        "profiled": manifest.get("profile"),
        "golden_sha256": manifest.get("golden_sha256"),
        "token_regression": golden_status(directory, manifest.get("golden_sha256")),
        "selected_range": manifest.get("range"), "aic_metrics": manifest.get("aic_metrics"),
        "cases": [],
    }
    for case in manifest["cases"]:
        case_dir = directory / f"case-{case['id']}"
        path = case_dir / "summary.json"
        if not path.is_file():
            report["cases"].append({"case": case, "ok": False, "error": "missing case summary"})
            continue
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry["server"] = server_stats(case_dir)
        if manifest.get("operator_capture"):
            try:
                hotspots = hotspot_summary(case_dir)
                entry["device_hotspots"] = hotspots
                entry["ok"] = entry["ok"] and hotspots["ok"]
                if hotspots.get("expected_ranks"):
                    entry["communication"] = communication_summary(case_dir / "traces", hotspots["expected_ranks"])
            except (KeyError, OSError, ValueError) as exc:
                entry.update({"ok": False, "device_hotspots": {"ok": False, "error": str(exc)}})
            capture = entry.get("operator_profile", {})
            if len(capture.get("requested_groups", [])) > 1:
                try:
                    extra = hotspot_summary(case_dir / "pmu-memory")
                except (KeyError, OSError, ValueError) as exc:
                    extra = {"ok": False, "error": str(exc)}
                entry["memory_replay_hotspots"] = extra
                entry["ok"] = entry["ok"] and extra["ok"]
        report["cases"].append(entry)
    report["ok"] = (completed and report["token_regression"] in ("unchecked", "passed")
                    and all(case["ok"] for case in report["cases"]))
    return report


def packaged_run(root):
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    manifests = sorted((root / "bench").glob("run-*/manifest.json"))
    runs = []
    for path in manifests:
        try:
            runs.append(summarize(path))
        except (KeyError, OSError, ValueError) as exc:
            runs.append({"directory": str(path.parent), "ok": False, "completed": False, "cases": [],
                         "error": f"incomplete result files: {exc}"})
    polls_path = root / "telemetry/polls.tsv"
    polls = []
    if polls_path.is_file():
        with polls_path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source, delimiter="\t"):
                polls.append((int(row["start_unix_ms"]), int(row["end_unix_ms"]), int(row["npu_smi_exit"])))
    if not runs:
        runs.append({"directory": str(root), "completed": False, "ok": False,
                     "cases": [], "error": "no benchmark manifest; inspect stderr.log"})
    for run in runs:
        run["host_run"] = str(root)
        run["process_status"] = status
        run["ok"] = run["ok"] and status["exit_code"] == 0
        if polls_path.is_file():
            run["device_sampling"] = {"completed_polls": len(polls), "failed_polls": sum(p[2] != 0 for p in polls),
                                      "polls_file": "telemetry/polls.tsv", "raw_file": "telemetry/samples.log"}
            for entry in run["cases"]:
                measurement = entry.get("measurement") or {}
                start, end = measurement.get("start_unix_ms"), measurement.get("end_unix_ms")
                if start is not None and end is not None:
                    entry["overlapping_device_polls"] = sum(
                        t0 <= end and t1 >= start and rc == 0 for t0, t1, rc in polls
                    )
        provenance = {}
        for key, name in (("selection", "selection.txt"), ("build", "build/build.txt"), ("image", "image.txt"),
                          ("command", "command.txt"), ("checkpoint", "checkpoint-metadata.sha256")):
            path = root / name
            if path.is_file():
                provenance[key] = path.read_text(encoding="utf-8").strip()
        for key, name in (("runtime", "runtime-versions.json"), ("operator_smoke", "operator-smoke.json")):
            path = root / name
            if path.is_file():
                try:
                    provenance[key] = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    provenance[key] = {"error": f"incomplete {name}"}
        calibration = root / "hardware-calibration.json"
        if calibration.is_file():
            provenance["hardware_calibration"], rates = reference_rates(calibration)
            isolated = provenance.get("operator_smoke", {}).get("microbenchmark", {})
            copy_rate = rates.get(isolated.get("physical_device"), {}).get("stream_copy")
            if isolated.get("status") in {"ok", "partial"} and copy_rate:
                for shape in isolated.get("shapes", []):
                    measured = shape.get("effective_gb_s")
                    if isinstance(measured, (int, float)) and math.isfinite(measured) and measured >= 0:
                        shape["logical_gb_s_over_copy_reference"] = measured / copy_rate
        run["provenance"] = provenance
    return runs, manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="*", help="extracted result directories")
    parser.add_argument("--compare-cases", type=Path, nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="compare two completed case-N directories")
    parser.add_argument("--feedback", type=Path, metavar="RUN", help="write a portable summary of one host run")
    args = parser.parse_args()
    if args.feedback:
        if args.directories or args.compare_cases:
            parser.error("--feedback accepts a single host run")
        root = args.feedback.resolve()
        os.chdir(root.parent)
        runs, _ = packaged_run(Path(root.name))
        print(json.dumps({"schema_version": 1, "ok": all(run["ok"] for run in runs), "runs": runs},
                         indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    if args.compare_cases:
        if args.directories:
            parser.error("choose either --compare-cases or result directories")
        try:
            comparison = compare_cases(*args.compare_cases)
        except (KeyError, OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    if not args.directories:
        parser.error("provide result directories or --compare-cases")
    manifests, hosts = set(), set()
    for directory in args.directories:
        directory = directory.resolve()
        packages = {p.parent for p in directory.rglob("status.json")
                    if p.parent.name.startswith("bench-")}
        matches = list(directory.rglob("run-*/manifest.json"))
        if directory.name.startswith("run-") and (directory / "manifest.json").is_file():
            matches.append(directory / "manifest.json")
        if not matches and not packages:
            parser.error(f"no benchmark manifest or package status in {directory}")
        manifests.update(matches)
        hosts.update(packages)
    reports = []
    for host in sorted(hosts):
        runs, children = packaged_run(host)
        reports.extend(runs)
        manifests.difference_update(children)
    reports.extend(summarize(path) for path in sorted(manifests))
    json.dump({"runs": reports}, sys.stdout, indent=2, ensure_ascii=False, allow_nan=False)
    print()
    return 0 if all(run["ok"] for run in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
