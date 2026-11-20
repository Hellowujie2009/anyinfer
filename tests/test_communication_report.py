import json
import tempfile
import unittest
from pathlib import Path

from tools.communication_report import communication_summary


class CommunicationReportTests(unittest.TestCase):
    def test_deduplicates_total_rows_and_retains_partial_counter_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp)
            out = trace / "rank-0/ASCEND_PROFILER_OUTPUT"
            out.mkdir(parents=True)
            op = {"Communication Time Info": {"Wait Time(ms)": 2, "Elapse Time(ms)": 3},
                  "Communication Bandwidth Info": {"HCCS": {"Transit Size(MB)": 4, "Transit Time(ms)": 1}}}
            (trace / "rank-0/communication.json").write_text("raw CANN input")
            (out / "communication.json").write_text(json.dumps({
                "step0": {"collective": {"hcom_allReduce_0": op, "Total Op Info": op}},
                "step1": {"collective": {"hcom_allreduce_1": {"Communication Time Info": {"Wait Time(ms)": "N/A"}}}},
            }))
            report = communication_summary(trace, 2)
            self.assertEqual(report["ranks_with_measurements"], 1)
            row, = report["ranks"][0]["operations"]
            self.assertEqual((row["records"], row["operation"]), (2, "allreduce"))
            self.assertEqual(row["times_ms"]["wait"], {"sum": 2, "records": 1})
            self.assertEqual(row["transport"]["HCCS"]["transit_mb"], {"sum": 4, "records": 1})
            self.assertEqual(report["ranks"][1]["status"], "not_exported")
