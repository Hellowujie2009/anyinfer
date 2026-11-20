"""Check trace attribution using fixture exports. No fixture duration is NPU benchmark data."""

import csv
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from tools import operator_report, report


class OperatorReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def rank(self, trace, rank):
        directory = trace / f"rank-{rank}"
        directory.mkdir(parents=True)
        (directory / "rank.json").write_text(json.dumps({"global_rank": rank, "status": "exported"}))
        return directory

    def csv(self, directory, durations):
        with (directory / "kernel_details.csv").open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["Name", "Type", "Duration(us)", "Input Shapes", "Input Data Types",
                             "Accelerator Core", "aic_mac_ratio", "Start Time(us)", "Stream ID"])
            for i, duration in enumerate(durations):
                writer.writerow(["flash_core", "SparseFlashMla", duration, "[2,128]", "BF16",
                                 "AI_CORE", 0.5, 100 * i, "0"])

    def test_rank_groups_counters_shape_and_no_double_counting(self):
        trace = self.root / "traces"
        index = self.root / "index.json"
        index.write_text(json.dumps({"tree_sha256": "example", "symbols": {"SparseFlashMla": ["local"]}}))
        for rank, durations in enumerate([[5, 10], [15]]):
            directory = self.rank(trace, rank)
            self.csv(directory, durations)
            (directory / "op_summary_duplicate.csv").write_text("unrelated inferior export")
        result = operator_report.summarize(trace, 2, index)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["sum_device_us"], 30)
        row, = result["operators"]
        self.assertEqual((row["calls"], row["p50_us"], row["p95_us"]), (3, 10, 15))
        self.assertEqual(row["rank_sum_device_us"], {0: 15, 1: 15})
        self.assertEqual(row["input_shapes"], "[2,128]")
        self.assertEqual(row["hardware_counters"]["aic_mac_ratio"], {"samples": 3, "mean": .5})
        self.assertEqual(row["source_candidates_by_exact_symbol"]["SparseFlashMla"], ["local"])
        self.assertEqual(result["same_stream_neighbors"][0]["sum_gap_us"], 95)

    def test_incomplete_capture_is_reported(self):
        directory = self.rank(self.root, 0)
        self.csv(directory, [])
        result = operator_report.summarize(self.root, 2, None)
        self.assertFalse(result["ok"])
        self.assertTrue(any("rank 1" in error for error in result["errors"]))
        self.assertTrue(any("no positive device duration" in error for error in result["errors"]))

    def test_reference_matches_physical_device_and_weights_pmu_by_event_duration(self):
        calibration = self.root / "hardware-calibration.json"
        calibration.write_text(json.dumps({"schema_version": 1, "status": "partial", "devices": [
            {"physical_device": 3, "cube_core_num": 4, "status": "ok", "probes": [
                {"id": "bf16_gemm", "reference_eligible": True, "status": "ok", "dtype": "bfloat16",
                 "measured_tflops": 0.000001},
                {"id": "stream_copy", "reference_eligible": True, "status": "ok", "dtype": "bfloat16",
                 "measured_gb_s": 10},
                {"id": "stream_copy", "reference_eligible": False, "status": "ok", "dtype": "bfloat16",
                 "measured_gb_s": 100}]}]}))
        for rank in range(2):
            d = self.rank(self.root, rank)
            (d / "rank.json").write_text(json.dumps({"rank": rank, "status": "exported",
                "physical_device": [3, 1][rank], "aic_metrics": "PipeUtilization"}))
            (d / "kernel_details.csv").write_text("Name,Type,Duration(us)\nmm,MatMul,100\n")
            with (d / "op_summary_1.csv").open("w", newline="") as out:
                writer = csv.writer(out)
                writer.writerow(["Op Name", "OP Type", "Task Duration(us)", "Task Type", "Input Shapes",
                    "Input Data Types", "Input Formats", "Output Shapes", "Output Data Types", "Output Formats",
                    "Task Start Time(us)", "Stream ID", "aic_mac_ratio", "Block Num"])
                for start, duration, stream, ratio in ((0, 8, 0, 0.25), (2, 4, 1, 1.0), (10, 2, 0, "N/A")):
                    writer.writerow(["mm", "MatMul", duration, "AI_CORE", "2,3;4,3", "BF16;BF16", "ND;ND",
                                     "2,4", "BFLOAT16", "ND", start, stream, ratio, 2])
        result = operator_report.summarize(self.root, 2, None, calibration)
        self.assertTrue(result["ok"], result["errors"])
        row, = result["operators"]
        self.assertEqual(row["calls"], 6)
        self.assertEqual(row["work_model"]["dense_flops_per_call"], 48)
        self.assertEqual(row["work_model"]["logical_tensor_bytes_per_call"], 52)
        r = row["rank_evidence"][0]
        self.assertEqual(r["duration_weighted_pmu"]["aic_mac_ratio"], {"mean": .5, "samples": 2, "covered_task_us": 12})
        self.assertAlmostEqual(r["logical_gb_s_over_copy_reference"], 52 * 3 / 14 / 1000 / 10)
        self.assertAlmostEqual(r["launch_geometry"]["blocks_per_physical_core"], 0.5)
        self.assertAlmostEqual(r["empirical_roofline"]["achieved_over_reference"], 48 * 3 / 14)
        self.assertNotIn("empirical_roofline", row["rank_evidence"][1])
        self.assertEqual(result["reference_comparison_coverage"]["fraction_of_summed_task_time"], .5)
        t, = result["ranks"][0]["timelines"]
        self.assertEqual((t["first_to_last_task_us"], t["task_union_us"], t["concurrent_task_us"],
                          t["longest_no_listed_task_us"]), (12, 10, 4, 2))

    def test_work_model_accepts_hcpost_and_skips_opaque_and_blocked_kernels(self):
        hc = ["HcPost", "AI_VECTOR_CORE", "[[1,3,8],[1,3,4,8],[1,3,4],[1,3,4,4]]",
              "DT_BF16;DT_BF16;FLOAT;FLOAT", "ND;ND;ND;ND", "8", "[1,3,4,8]", "BF16", "ND"]
        self.assertEqual(operator_report.work_model(hc), {"logical_tensor_bytes_per_call": 672})
        for attr, value in ((0, "SparseFlashMla"), (1, "MIX_AIV"), (3, "INT8;BF16;FLOAT;FLOAT"),
                            (4, "FRACTAL_NZ;ND;ND;ND"), (6, "N/A")):
            changed = hc.copy()
            changed[attr] = value
            self.assertIsNone(operator_report.work_model(changed))

    def test_opaque_memory_tasks_compare_only_identical_pmu_columns_in_one_core_domain(self):
        directory = self.rank(self.root, 0)
        (directory / "rank.json").write_text(json.dumps({"rank": 0, "status": "exported", "aic_metrics": "Memory"}))
        with (directory / "kernel_details.csv").open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["Type", "Task Type", "Duration(us)", "aic_main_mem_read_bw(GB/s)"])
            for op, core, rate, n in (("RouteA", "AI_CORE", 4, 3), ("RouteB", "AI_CORE", 10, 3),
                                      ("Vector", "AI_VECTOR_CORE", 90, 3), ("Singleton", "AI_CORE", 99, 1)):
                for _ in range(n):
                    writer.writerow([op, core, 2, rate])
        result = operator_report.summarize(self.root, 1, None)
        self.assertTrue(result["ok"], result["errors"])
        top, = result["ranks"][0]["observed_main_memory_rates"]
        self.assertEqual((top["op_type"], top["eligible_groups"]), ("RouteB", 2))
        op = next(row for row in result["operators"] if row["op_type"] == "RouteA")
        self.assertIsNone(op["work_model"])
        counter = op["rank_evidence"][0]["duration_weighted_pmu"][top["counter"]]
        self.assertAlmostEqual(counter["over_highest_observed_group"], .4)

    def test_requested_memory_group_requires_numeric_bandwidth_on_each_active_rank(self):
        for rank in range(2):
            directory = self.rank(self.root, rank)
            marker = directory / "rank.json"
            marker.write_text(json.dumps({"rank": rank, "status": "exported", "aic_metrics": "Memory"}))
            export = directory / "kernel_details.csv"
            header = "Name,Type,Duration(us),Block Num,aic_main_mem_read_bw(GB/s),main_mem_write_bw(GB/s)\n"
            export.write_text(header + "inner,Compressor,5,16,NaN,N/A\n")
        export.write_text(header + "inner,Compressor,5,16,0,2.5\n")
        result = operator_report.summarize(self.root, 2, None)
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"], ["rank 0: requested Memory PMU group produced no memory counters"])
        row, = result["operators"]
        self.assertEqual(row["block_dim"], "16")
        self.assertEqual(row["hardware_counters"]["aic_main_mem_read_bw(GB/s)"], {"samples": 1, "mean": 0})
        self.assertEqual(row["hardware_counters"]["main_mem_write_bw(GB/s)"], {"samples": 1, "mean": 2.5})
        (self.root / "rank-0" / export.name).write_text(export.read_text())
        self.assertTrue(operator_report.summarize(self.root, 2, None)["ok"])

    def test_database_timestamps_converted_to_microseconds(self):
        directory = self.rank(self.root, 0)
        with closing(sqlite3.connect(directory / "msprof_test.db")) as db:
            db.executescript("""
                CREATE TABLE TASK(globalTaskId,deviceId,streamId,startNs,endNs);
                CREATE TABLE COMPUTE_TASK_INFO(globalTaskId,name,opType,taskType,inputShapes,inputDataTypes,blockDim);
                CREATE TABLE STRING_IDS(id,value);
                INSERT INTO STRING_IDS VALUES(1,'kernel'),(2,'Compressor'),(3,'AI_CORE');
                INSERT INTO TASK VALUES(1,0,8,2000000,2025000);
                INSERT INTO COMPUTE_TASK_INFO VALUES(1,1,2,3,NULL,NULL,24);
            """)
        result = operator_report.summarize(self.root, 1, None)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["sum_device_us"], 25)
        self.assertEqual(result["operators"][0]["op_type"], "Compressor")
        self.assertIsNone(result["operators"][0]["input_shapes"])

    def test_compare_keeps_disappearing_operators_and_rejects_replay_changes(self):
        cases = []
        for label, op in (("base", "MatMul"), ("new", "FusedMatMul")):
            path = self.root / label / "bench" / "run-one" / "case-1"
            path.mkdir(parents=True)
            cases.append(path)
            (self.root / label / "selection.txt").write_text("hardware=a2\n")
            (path.parent / "manifest.json").write_text(json.dumps({"prompt_generator": "a"}))
            summary = {"case": {"id": 1}, "operator_profile": {"replay": {"requests": 4}},
                       "ok": True, "measurement": {"successful_requests": 4}}
            (path / "summary.json").write_text(json.dumps(summary))
            (path / "operators.json").write_text(json.dumps({"ok": True, "expected_ranks": 2,
                "operators": [{"op_type": op, "sum_device_us": 7, "calls": 2}]}))
        comparison = report.compare_cases(*cases)
        rows = comparison["operator_replay_deltas"]
        self.assertIn("supplied_weight_manifest_sha256", comparison["unverified_environment_fields"])
        self.assertEqual({r["op_type"]: r["delta_sum_device_us"] for r in rows}, {"FusedMatMul": 7, "MatMul": -7})
        (self.root / "new/selection.txt").write_text("hardware=a3\n")
        with self.assertRaisesRegex(ValueError, "hardware_family"):
            report.compare_cases(*cases)
        (self.root / "new/selection.txt").write_text("hardware=a2\n")
        for index, path in enumerate(cases):
            (path.parents[2] / "checkpoint-metadata.sha256").write_text(str(index))
        with self.assertRaisesRegex(ValueError, "checkpoint_metadata_sha256"):
            report.compare_cases(*cases)
        (self.root / "new/checkpoint-metadata.sha256").write_text("0")
        for index, path in enumerate(cases):
            operator_file = path / "operators.json"
            payload = json.loads(operator_file.read_text())
            payload["ranks"] = [{"rank": r, "worker": {"dp_rank": r * index, "tp_rank": r * (1 - index)}} for r in range(2)]
            operator_file.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "rank_layout"):
            report.compare_cases(*cases)
        summary["operator_profile"]["replay"]["requests"] = 1
        (cases[1] / "summary.json").write_text(json.dumps(summary))
        with self.assertRaises(ValueError):
            report.compare_cases(*cases)

    def test_portable_feedback_reports_device_poll_coverage_gauges_and_missing_golden(self):
        root = self.root / "bench-a2-SHARD"
        directory = root / "bench/run-0"
        case = directory / "case-100"
        case.mkdir(parents=True)
        (root / "telemetry").mkdir()
        (root / "telemetry/polls.tsv").write_text("start_unix_ms\tend_unix_ms\tnpu_smi_exit\n0\t3\t0\n"
                                                 "50\t61\t0\n60\t92\t124\n")
        (root / "status.json").write_text(json.dumps({"exit_code": 1}))
        (root / "selection.txt").write_text("range=100-200")
        (directory / "manifest.json").write_text(json.dumps({"suite_sha256": "suite", "range": "100-200",
            "golden_sha256": "missing", "cases": [{"id": 100}, {"id": 101}]}))
        (case / "summary.json").write_text(json.dumps({"case": {"id": 100}, "ok": True,
            "measurement": {"start_unix_ms": 60, "end_unix_ms": 85}}))
        with (case / "engine-gauges.jsonl").open("w") as out:
            for value in (4, 9, 2):
                line = 'vllm:num_requests_running{engine="0"} ' + str(value)
                out.write(json.dumps({"unix_ms": 60 + value, "metrics": [line]}) + "\n")
        cmd = [sys.executable, str(Path(report.__file__).resolve()), "--feedback", str(root)]
        reply = json.loads(subprocess.check_output(cmd))
        self.assertFalse(reply["ok"])
        result, = reply["runs"]
        self.assertEqual(result["host_run"], root.name)
        self.assertEqual(result["token_regression"], "incomplete")
        self.assertEqual(result["provenance"]["selection"], "range=100-200")
        self.assertEqual(result["device_sampling"]["failed_polls"], 1)
        self.assertEqual(result["cases"][0]["overlapping_device_polls"], 1)
        gauges = result["cases"][0]["server"]["gauge_polls"]
        self.assertEqual(gauges["peak_by_series"], {'vllm:num_requests_running{engine="0"}': 9})
        self.assertFalse(result["cases"][1]["ok"])

    def test_two_pmu_replays_keep_separate_times_and_reports(self):
        case = self.root / "case-7"
        extra = case / "pmu-memory"
        extra.mkdir(parents=True)
        mf = self.root / "manifest.json"
        mf.write_text(json.dumps({"operator_capture": "separate_replay", "cases": [{"id": 7}], "suite_sha256": "suite"}))
        (self.root / "summary.json").write_text("{}")
        (case / "summary.json").write_text(json.dumps({"ok": True, "case": {"id": 7},
            "operator_profile": {"requested_groups": ["PipeUtilization", "Memory"]}}))
        for directory, total in [(case, 12), (extra, 15)]:
            (directory / "operators.json").write_text(json.dumps({"ok": True, "operators": [
                {"op_type": "Compressor", "calls": 2, "sum_device_us": total}]}))
        entry, = report.summarize(mf)["cases"]
        self.assertEqual(entry["device_hotspots"]["top_by_sum_device_us"][0]["sum_device_us"], 12)
        self.assertEqual(entry["memory_replay_hotspots"]["top_by_sum_device_us"][0]["sum_device_us"], 15)
        (extra / "operators.json").unlink()
        entry, = report.summarize(mf)["cases"]
        self.assertFalse(entry["ok"])
        self.assertTrue(entry["device_hotspots"]["ok"])


if __name__ == "__main__":
    unittest.main()
