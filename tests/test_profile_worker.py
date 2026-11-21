import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class ProfileWorkerTests(unittest.TestCase):
    def test_two_pmu_exports_have_independent_rank_paths_and_unsupported_pmu_can_stop(self):
        npu = MagicMock()
        npu.current_device.return_value = 0
        profiler = MagicMock()
        profiler.AiCMetrics = SimpleNamespace(PipeUtilization="pipe", Memory="memory")
        profiler.supported_ai_core_metrics.return_value = {"pipe", "memory"}
        group = lambda: SimpleNamespace(rank_in_group=0)
        dependencies = {
            "torch": SimpleNamespace(npu=npu), "torch_npu": SimpleNamespace(profiler=profiler, __version__="test"),
            "vllm.distributed.parallel_state": SimpleNamespace(get_dp_group=group, get_tp_group=group),
            "vllm_ascend.worker.worker": SimpleNamespace(NPUWorker=object),
        }
        path = Path(__file__).resolve().parents[1] / "reference/runtime/bench_worker.py"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, dependencies), patch.dict(
            os.environ, ANYINFER_BENCH_RUN=tmp, ASCEND_RT_VISIBLE_DEVICES="3"
        ):
            spec = importlib.util.spec_from_file_location("bench_worker_fixture", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            worker = module.BenchmarkWorker()
            worker.rank, worker.parallel_config = 0, SimpleNamespace(tensor_parallel_size=1)
            for case, name, secondary, supported in ((1, "PipeUtilization", False, True),
                    (1, "Memory", True, True), (2, "Memory", False, False), (3, "PipeUtilization", False, True)):
                if not supported:
                    profiler.supported_ai_core_metrics.return_value = {"pipe"}
                root = Path(tmp) / f"case-{case}"
                if secondary:
                    root /= "pmu-memory"
                trace = root / "traces"
                trace.mkdir(parents=True)
                (Path(tmp) / "active-case.json").write_text(json.dumps({
                    "case_id": case, "aic_metrics": name, "secondary_memory": secondary}))
                worker.profile(True)
                worker.profile(False)
                record = json.loads((trace / "rank-0/rank.json").read_text())
                self.assertEqual(record["status"], "exported" if supported else "unsupported")
                self.assertEqual((record["case_id"], record["physical_device"], record["aic_metrics"]), (case, 3, name))
            self.assertEqual(profiler.profile.return_value.stop.call_count, 3)
