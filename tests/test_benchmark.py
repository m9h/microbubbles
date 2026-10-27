"""CPU checks for the CUDA benchmark protocol, not GPU performance claims."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from ultratrace_ulm import benchmark as bench
from tests.helpers import GRID


class BenchmarkTests(unittest.TestCase):
    def test_fixture_is_repeatable_and_records_full_shape(self):
        spec = bench.Fixture(frames=5, angles=3, rows=2, cols=4, seed=17)
        first = bench.make_fixture(spec)
        second = bench.make_fixture(spec)
        self.assertEqual(first[0].shape, (5, 3, 2, 4, 512))
        self.assertEqual(first[0].dtype, np.complex64)
        self.assertTrue(first[0].flags.c_contiguous)
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(first[1].shape, (3, 4))
        self.assertEqual(first[2].shape, (3, 2))
        meta = bench.fixture_metadata(spec, first)
        self.assertEqual(meta["iq_shape"], [5, 3, 2, 4, 512])
        self.assertEqual(meta["config"]["sampling_rate_hz"], 20e6)
        self.assertEqual(len(meta["iq_sha256"]), 64)

    def test_invalid_options_fail_before_workers_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "report.json"
            for args in (["--chunks", "0"], ["--frames", "0"],
                         ["--warmups", "0"], ["--repeats", "-1"],
                         ["--seed", "-1"], ["--angles", "-1"]):
                with self.subTest(args=args), patch.object(bench, "run_benchmark") as run, \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        bench.main(["--output", str(output), *args])
                    self.assertEqual(caught.exception.code, 2)
                    run.assert_not_called()
                    self.assertFalse(output.parent.exists())

    def test_pool_meter_counts_live_and_reserved_separately(self):
        pool = SimpleNamespace(used=0, reserved=0)

        def malloc(size):
            pool.used += size
            pool.reserved = max(pool.reserved, pool.used)
            return object()

        pool.malloc = malloc
        pool.used_bytes = lambda: pool.used
        pool.total_bytes = lambda: pool.reserved
        meter = bench.PoolMeter(pool)
        meter.malloc(100)
        pool.used = 0  # first allocation freed, still reserved
        meter.malloc(60)
        pool.reserved = 300  # cached/fragmented blocks exceed live allocations
        meter.malloc(80)
        pool.used = 0
        result = meter.snapshot()
        self.assertEqual(result["cupy_pool_live_peak_bytes"], 140)
        self.assertEqual(result["cupy_pool_reserved_peak_bytes"], 300)
        self.assertEqual(result["cupy_pool_live_after_bytes"], 0)
        self.assertEqual(result["cupy_pool_reserved_after_bytes"], 300)

    def test_timing_synchronizes_and_excludes_instrumented_memory_pass(self):
        spec = bench.Fixture(frames=2)
        events = []
        pool = Mock()
        pool.used_bytes.return_value = 0
        pool.total_bytes.return_value = 0

        @contextlib.contextmanager
        def allocator(fn):
            events.append("meter-on")
            yield
            events.append("meter-off")

        cp = SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(
                deviceSynchronize=lambda: events.append("sync")), using_allocator=allocator),
            get_default_memory_pool=lambda: pool,
        )

        def beamform(*args, **kwargs):
            events.append("beamform")
            self.assertEqual(kwargs["n_chunks"], 3)
            return np.ones((2, 3, 11, 17), dtype=np.complex64), GRID

        with patch.object(bench, "_cuda_module", return_value=cp), \
             patch.object(bench.core, "beamform_iq", side_effect=beamform), \
             patch.object(bench.time, "perf_counter", side_effect=[0, 4, 10, 12, 20, 23]):
            result, output = bench.measure_case(spec, n_chunks=3, warmups=1, repeats=2)
        self.assertEqual(result["warmup_seconds"], [4])
        self.assertEqual(result["repeat_seconds"], [2, 3])
        self.assertEqual(result["median_seconds"], 2.5)
        self.assertEqual(result["effective_chunks"], 2)
        self.assertEqual(events[:9], ["sync", "beamform", "sync"] * 3)
        self.assertEqual(events[9:], ["sync", "meter-on", "beamform", "sync", "meter-off"])
        pool.free_all_blocks.assert_called_once()
        self.assertGreater(result["memory"]["process_peak_rss_bytes"], 0)
        self.assertEqual(output.dtype, np.complex64)

    def test_reference_sums_host_angles_with_one_frame_on_device(self):
        spec = bench.Fixture(frames=2, angles=3)
        angles = [np.full((GRID.x.size, 1), i + 1j, dtype=np.complex64) for i in (1, 2, 3, 4, 5, 6)]
        expected = np.stack([np.full((3, 11, 17), 6 + 3j), np.full((3, 11, 17), 15 + 3j)]).astype(np.complex64)

        def beamform(*args, **kwargs):
            self.assertEqual(kwargs["n_chunks"], spec.frames)
            for _ in angles:
                bench.core._mach_beamform(channel_data=None)
            return expected.copy(), GRID

        kernel = Mock(side_effect=angles)
        with patch.object(bench, "_cuda_module", return_value=SimpleNamespace(asnumpy=np.array)), \
             patch.object(bench.core, "_mach_beamform", kernel), \
             patch.object(bench.core, "beamform_iq", side_effect=beamform):
            actual = bench.host_reference(spec)
            self.assertIs(bench.core._mach_beamform, kernel)
        np.testing.assert_array_equal(actual, expected)

    def test_cli_writes_failure_report_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "failure.json"
            failure = {"status": "failed", "reference": {"error": "CUDA unavailable"}}
            with patch.object(bench, "run_benchmark", return_value=failure), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = bench.main(["--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.read_text()), failure)
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_comparison_rejects_invalid_or_degenerate_outputs(self):
        reference = np.ones((2, 1, 2, 3), dtype=np.complex64)
        good = bench.compare_outputs(reference.copy(), reference)
        self.assertTrue(good["passed"])
        self.assertEqual(good["max_absolute_error"], 0)
        for candidate in (reference * 2, reference[::-1, :, :, :2],
                          np.full_like(reference, np.nan), reference.astype(np.complex128)):
            with self.subTest(shape=candidate.shape, dtype=candidate.dtype):
                self.assertFalse(bench.compare_outputs(candidate, reference)["passed"])
        self.assertFalse(bench.compare_outputs(np.zeros_like(reference), np.zeros_like(reference))["passed"])

    def test_each_setting_uses_a_fresh_worker_and_same_reference(self):
        jobs = []

        def worker(request, directory):
            jobs.append(dict(request))
            data = np.ones((2, 1, 2, 3), dtype=np.complex64)
            path = Path(directory) / f"case-{len(jobs)}.npy"
            np.save(path, data)
            return {"output_path": str(path), "pid": len(jobs), "status": "ok"}

        with patch.object(bench, "_run_worker", side_effect=worker):
            report = bench.run_benchmark(bench.Fixture(frames=2), [1, 3, 7], warmups=1, repeats=2)
        self.assertEqual([j["mode"] for j in jobs], ["reference", "measure", "measure", "measure"])
        self.assertEqual([j["n_chunks"] for j in jobs[1:]], [1, 3, 7])
        self.assertEqual([r["pid"] for r in report["cases"]], [2, 3, 4])
        self.assertTrue(all(r["comparison"]["passed"] for r in report["cases"]))
        self.assertEqual(report["status"], "ok")
        json.dumps(report, allow_nan=False)

    def test_failed_case_is_recorded_and_later_cases_still_run(self):
        def worker(request, directory):
            if request.get("n_chunks") == 1 and request["mode"] == "measure":
                raise RuntimeError("CUDA out of memory")
            path = Path(directory) / f"{request['mode']}-{request.get('n_chunks')}.npy"
            np.save(path, np.ones((2, 1, 2, 3), dtype=np.complex64))
            return {"output_path": str(path), "status": "ok"}

        with patch.object(bench, "_run_worker", side_effect=worker):
            report = bench.run_benchmark(bench.Fixture(frames=2), [1, 2], warmups=1, repeats=1)
        self.assertEqual(report["status"], "failed")
        self.assertIn("out of memory", report["cases"][0]["error"])
        self.assertTrue(report["cases"][1]["comparison"]["passed"])

    def test_worker_actually_launches_another_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            request = {"mode": "invalid"}
            with self.assertRaisesRegex(RuntimeError, "Unknown worker mode"):
                bench._run_worker(request, Path(directory))

    @unittest.skipUnless(os.environ.get("ULM_TEST_CUDA") == "1", "set ULM_TEST_CUDA=1 on a CUDA host")
    def test_real_cuda_benchmark_smoke(self):
        report = bench.run_benchmark(bench.Fixture(frames=3), [1, 2, 5], warmups=1, repeats=2)
        self.assertEqual(report["status"], "ok", report)
        pids = [report["reference"]["pid"], *[r["pid"] for r in report["cases"]]]
        self.assertEqual(len(pids), len(set(pids)))
        for case in report["cases"]:
            self.assertTrue(case["comparison"]["passed"])
            self.assertGreater(case["memory"]["cupy_pool_live_peak_bytes"], 0)
            self.assertGreaterEqual(case["memory"]["cupy_pool_reserved_peak_bytes"],
                                    case["memory"]["cupy_pool_live_peak_bytes"])


if __name__ == "__main__":
    unittest.main()
