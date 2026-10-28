"""The benchmark worker must create the CUDA context BEFORE building the host fixture.

On a DGX Spark (GB10, unified memory, THP=madvise, CUDA 13.0 / cupy 14.2) the FIRST
device allocation fails intermittently -- "Out of memory allocating 3,072 bytes (allocated so
far: 0 bytes)" -- when a >= 1 GiB NumPy array (madvise'd to huge pages) was allocated and
touched just before it (2 of 6 fresh processes; 0 of 6 with the device touched first, 0 of 6
with NUMPY_MADVISE_HUGEPAGE=0). The 128-frame "larger" fixture is 1.12 GiB, so the reference
worker died on its first cudaMalloc while the 16-frame fixture never did. Initialising the
device before `make_fixture` makes the worker order-independent of that platform quirk.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from ultratrace_ulm import benchmark as bench


class WorkerInitTests(unittest.TestCase):
    def _fake_cupy(self, events):
        def asarray(array, *args, **kwargs):
            events.append("device")
            return np.asarray(array)
        return SimpleNamespace(
            asarray=Mock(side_effect=asarray),
            cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=Mock(side_effect=lambda: events.append("sync")))),
            get_default_memory_pool=Mock(return_value=SimpleNamespace(used_bytes=lambda: 0, total_bytes=lambda: 0,
                                                                       free_all_blocks=lambda: None)),
        )

    def test_reference_worker_touches_the_device_before_building_the_fixture(self):
        events = []
        cp = self._fake_cupy(events)
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "req.json"; response = Path(tmp) / "res.json"
            request.write_text(json.dumps({"mode": "reference", "fixture": {
                "frames": 2, "angles": 1, "rows": 1, "cols": 2, "samples": 8, "elev_planes": 1, "seed": 0}}))
            with patch.object(bench, "_cuda_module", return_value=cp), \
                 patch.object(bench, "_environment", return_value={}), \
                 patch.object(bench, "make_fixture", side_effect=lambda spec: events.append("fixture") or (None,)) , \
                 patch.object(bench, "host_reference", side_effect=lambda spec: (bench.make_fixture(spec),
                                                                                 np.ones((2, 1, 1, 1), np.complex64))[1]):
                bench._worker(str(request), str(response))
        self.assertIn("device", events, "worker never touched the device before the fixture")
        self.assertLess(events.index("device"), events.index("fixture"),
                        f"device must be initialised before the fixture is built; order was {events}")

    def test_measure_worker_touches_the_device_before_building_the_fixture(self):
        events = []
        cp = self._fake_cupy(events)
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "req.json"; response = Path(tmp) / "res.json"
            request.write_text(json.dumps({"mode": "measure", "n_chunks": 1, "warmups": 1, "repeats": 1,
                                           "fixture": {"frames": 2, "angles": 1, "rows": 1, "cols": 2,
                                                       "samples": 8, "elev_planes": 1, "seed": 0}}))
            fake_result = ({"n_chunks": 1}, np.ones((2, 1, 1, 1), np.complex64))
            with patch.object(bench, "_cuda_module", return_value=cp), \
                 patch.object(bench, "_environment", return_value={}), \
                 patch.object(bench, "make_fixture", side_effect=lambda spec: events.append("fixture") or (None,)), \
                 patch.object(bench, "measure_case", side_effect=lambda spec, **kw: (bench.make_fixture(spec), fake_result)[1]):
                bench._worker(str(request), str(response))
        self.assertIn("device", events)
        self.assertLess(events.index("device"), events.index("fixture"), f"order was {events}")
