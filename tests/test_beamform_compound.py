"""Composition contract on CPU; opt-in real MACH/CuPy comparison on CUDA."""

import os
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

from ultratrace_ulm import beamform_core as core
from tests.helpers import CONFIG, GRID


class CompoundTests(unittest.TestCase):
    def test_angle_sum_layout_dtype_and_single_host_transfer(self):
        cp = ModuleType("cupy")
        cp.__dict__.update({k: v for k, v in np.__dict__.items() if not k.startswith("__")})
        def to_host(array, out=None):
            if out is None:
                return np.array(array)
            np.copyto(out, array)
            return out

        cp.asnumpy = Mock(side_effect=to_host)
        rng = np.random.default_rng(7)
        config = core.NeutralConfig.from_h5_attrs(CONFIG)
        for angles in (1, 2, 4, 9):
            for frames in (1, 5):
                with self.subTest(angles=angles, frames=frames):
                    outputs = (rng.normal(size=(angles, GRID.x.size, frames)) +
                               1j * rng.normal(size=(angles, GRID.x.size, frames))).astype(np.complex64)
                    iq = np.ones((frames, angles, 2, 4, 8), dtype=np.complex64)
                    cp.asnumpy.reset_mock()
                    with patch.dict(sys.modules, {"cupy": cp}), \
                         patch.object(core, "MACH_AVAILABLE", True), \
                         patch.object(core, "build_grid", return_value=GRID), \
                         patch.object(core, "_mach_beamform", side_effect=list(outputs)):
                        result, grid = core.beamform_iq(iq, np.zeros((angles, 4)), np.zeros((angles, 2)), config)
                    expected = outputs.sum(axis=0).reshape(*GRID.x.shape, frames).transpose(3, 1, 0, 2)
                    np.testing.assert_array_equal(result, expected)
                    self.assertIs(grid, GRID)
                    self.assertEqual(result.dtype, np.complex64)
                    self.assertTrue(result.flags.c_contiguous)
                    cp.asnumpy.assert_called_once()

    @unittest.skipUnless(os.environ.get("ULM_TEST_CUDA") == "1", "set ULM_TEST_CUDA=1 on a CUDA host")
    def test_real_cuda_matches_host_compounding(self):
        import cupy as cp

        self.assertTrue(core.MACH_AVAILABLE, "Install a mach-cuda extra before running the CUDA test")
        config = core.NeutralConfig.from_h5_attrs(CONFIG)
        rng = np.random.default_rng(3)
        iq = (rng.normal(size=(3, 4, 2, 4, 512)) +
              1j * rng.normal(size=(3, 4, 2, 4, 512))).astype(np.complex64)
        kernel = core._mach_beamform
        angle_outputs = []

        def record_angle(**kwargs):
            result = kernel(**kwargs)
            angle_outputs.append(cp.asnumpy(result))
            return result

        with patch.object(core, "_mach_beamform", side_effect=record_angle):
            result, grid = core.beamform_iq(iq, np.zeros((4, 4)), np.zeros((4, 2)), config)
        expected = np.stack(angle_outputs).sum(axis=0).reshape(*grid.x.shape, 3).transpose(3, 1, 0, 2)
        self.assertTrue(np.any(np.abs(expected) > 0), "Fixture must exercise nonzero beamforming")
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-6)
        for count in (2, 5):
            with self.subTest(n_chunks=count):
                chunked, _ = core.beamform_iq(
                    iq, np.zeros((4, 4)), np.zeros((4, 2)), config, n_chunks=count,
                )
                np.testing.assert_allclose(chunked, expected, rtol=2e-6, atol=1e-6)
