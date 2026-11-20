"""Case-scoped Ascend profiler for the pinned vLLM-Ascend benchmark worker."""

import json
import os
import time
from pathlib import Path

import torch
import torch_npu
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm_ascend.worker.worker import NPUWorker


class BenchmarkWorker(NPUWorker):
    # Inference and runner selection are inherited from the release worker.
    def profile(self, is_start=True, profile_prefix=None):
        if is_start:
            if getattr(self, "_case_profiler", None) is not None or getattr(self, "_unsupported_capture", False):
                raise RuntimeError("previous operator capture is still active")
            run = Path(os.environ["ANYINFER_BENCH_RUN"])
            config = json.loads((run / "active-case.json").read_text())
            case = config["case_id"]
            group = config["aic_metrics"]
            dp = get_dp_group().rank_in_group
            tp = get_tp_group().rank_in_group
            rank = dp * self.parallel_config.tensor_parallel_size + tp
            ordinal = torch.npu.current_device()
            devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            case_root = run / f"case-{case}"
            if config.get("secondary_memory"):
                if group != "Memory":
                    raise ValueError(f"unexpected secondary PMU group: {group}")
                case_root = case_root / "pmu-memory"
            root = case_root / "traces" / f"rank-{rank}"
            root.mkdir(exist_ok=False)
            self._case_rank_file = root / "rank.json"
            self._case_rank = dict(case_id=case, rank=rank, global_rank=rank, dp_rank=dp, tp_rank=tp,
                                   worker_rank=self.rank, device_ordinal=ordinal,
                                   physical_device=int(devices[ordinal]), pid=os.getpid(),
                                   torch_npu_version=torch_npu.__version__,
                                   status="starting", aic_metrics=group, start_unix_ns=time.time_ns())
            self._case_rank_file.write_text(json.dumps(self._case_rank))
            profiler = torch_npu.profiler
            metric = getattr(profiler.AiCMetrics, group, None)
            if metric is None or metric not in profiler.supported_ai_core_metrics():
                self._unsupported_capture = True
                self._case_rank.update(status="unsupported", error=f"installed torch_npu does not support {group}")
                self._case_rank_file.write_text(json.dumps(self._case_rank))
                return
            options = profiler._ExperimentalConfig(
                export_type=profiler.ExportType.Text,
                profiler_level=profiler.ProfilerLevel.Level1,
                aic_metrics=metric,
                data_simplification=False,
            )
            self._case_profiler = profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                record_shapes=True,
                with_stack=False,
                experimental_config=options,
                on_trace_ready=profiler.tensorboard_trace_handler(
                    str(root), worker_name=f"rank{rank}", analyse_flag=True, async_mode=False
                ),
            )
            torch.npu.synchronize()
            self._case_profiler.start()
            return
        if getattr(self, "_unsupported_capture", False):
            self._unsupported_capture = False
            return
        if getattr(self, "_case_profiler", None) is None:
            raise RuntimeError("no active operator capture")
        try:
            torch.npu.synchronize()
            self._case_profiler.stop()
            self._case_rank.update(status="exported", stop_unix_ns=time.time_ns())
            self._case_rank_file.write_text(json.dumps(self._case_rank))
        finally:
            self._case_profiler = None
