"""Clock units, calibration coverage and persistence, using simulated devices only."""

import contextlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deploy import calibrate


class CalibrationTests(unittest.TestCase):
    def test_timing_uses_per_call_device_milliseconds(self):
        clock = 0

        class Event:
            def __init__(self, enable_timing):
                self.enabled = enable_timing

            def record(self):
                self.tick = clock

            def synchronize(self):
                pass

            def elapsed_time(self, end):
                return end.tick - self.tick

        def step():
            nonlocal clock
            clock += 2

        npu = SimpleNamespace(Event=Event, synchronize=lambda: None)
        self.assertEqual(calibrate.event_samples_ms(npu, step, 2, 3, 4), [2, 2, 2])
        self.assertEqual(clock, (2 + 3 * 4) * 2)
        with self.assertRaisesRegex(ValueError, "invalid NPU event duration"):
            calibrate.event_samples_ms(npu, lambda: None)

    def test_rates_include_copy_read_and_write_and_gemm_multiply_add(self):
        samples = [2.0, 1.0, 20.0]
        probe = calibrate.probe_definition("stream_copy", 256)
        self.assertEqual(probe["logical_bytes_per_call"], 512 * 1024**2)
        self.assertEqual(calibrate.summarize_samples(probe, samples)["measured_gb_s"], 512 * 1024**2 / 2e6)
        self.assertFalse(calibrate.probe_definition("stream_copy", 32)["reference_eligible"])
        for kind, field in (("bf16_gemm", "measured_tflops"), ("int8_quant_gemm", "measured_tops")):
            probe = calibrate.probe_definition(kind, 4096)
            rate = calibrate.summarize_samples(probe, samples)
            self.assertTrue(math.isclose(rate[field], 2 * 4096**3 / 2e9))

    def test_probe_failure_retains_other_rates_and_physical_device_order(self):
        npu = MagicMock()
        npu.device_count.return_value = 2
        npu.get_device_name.side_effect = ["fake-device-0", "fake-device-1"]
        npu.get_cann_version.return_value = "fixture"
        torch = SimpleNamespace(npu=npu, inference_mode=contextlib.nullcontext, __version__="fixture")

        def run_probe(module, dev, probe):
            if dev == "npu:1" and probe["id"] == "int8_quant_gemm":
                raise RuntimeError("fixture failure")
            return {**calibrate.summarize_samples(probe, [1.0]), "status": "ok"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hardware-calibration.json"
            with patch.dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="5,3", ANYINFER_HARDWARE="a3"), \
                 patch.dict(sys.modules, torch=torch, torch_npu=SimpleNamespace(__version__="fixture")), \
                 patch.object(calibrate, "run_probe", run_probe):
                result = calibrate.calibrate(path, "5,3")
            self.assertEqual(json.loads(path.read_text()), result)
            self.assertEqual(result["status"], "partial")
            self.assertEqual([d["physical_device"] for d in result["devices"]], [5, 3])
            self.assertEqual([d["ordinal"] for d in result["devices"]], [0, 1])
            self.assertEqual([d["status"] for d in result["devices"]], ["ok", "partial"])
            probes = result["devices"][1]["probes"]
            self.assertEqual(len(probes), 6)
            self.assertNotIn("measured_tops", probes[-1])
            self.assertEqual(probes[-1]["status"], "failed")
            self.assertEqual(probes[3]["status"], "ok")
            self.assertEqual(calibrate.calibrate(path, "5,3", skip=True)["status"], "skipped")
            self.assertEqual(calibrate.calibrate(path, "5,5")["status"], "unavailable")
            self.assertFalse(path.with_name(path.name + ".part").exists())
