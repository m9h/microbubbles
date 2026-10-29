"""The benchmark comparison must tolerate float32 accumulation round-off, not chunking.

On a GB10 (2026-10-27) every chunk setting of the "larger" fixture failed `compare_outputs`
with the IDENTICAL max-abs error 1.53e-5 and relative-L2 error 1.2e-7 -- float32 epsilon
(~6e-8) times a 9-angle sum, from device-side versus host-side accumulation order. A fixed
atol of 1e-6 cannot be honoured by complex64 on O(1) outputs, so the harness reported a
correct implementation as failed. The absolute tolerance must scale with the reference
magnitude; a real numerical error (1e-3 relative) must still fail.
"""

import unittest

import numpy as np

from ultratrace_ulm import benchmark as bench


def _reference(scale=5.0, seed=11, shape=(4, 3, 6, 7)):
    rng = np.random.default_rng(seed)
    return (scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))).astype(np.complex64)


class ToleranceTests(unittest.TestCase):
    def test_float32_roundoff_level_difference_passes(self):
        reference = _reference()
        # perturb at ~2 float32 ulps of the magnitude scale: what a different summation order gives
        delta = (np.float32(1.5e-5) * (np.random.default_rng(1).choice([-1, 1], size=reference.shape))).astype(np.complex64)
        candidate = (reference + delta).astype(np.complex64)
        result = bench.compare_outputs(candidate, reference)
        self.assertTrue(result["passed"], result)
        self.assertLess(result["relative_l2_error"], 1e-5)
        self.assertGreater(result["max_absolute_error"], bench.ATOL)   # would have failed under a fixed atol

    def test_real_numerical_error_still_fails(self):
        reference = _reference()
        candidate = (reference * np.float32(1 + 1e-3)).astype(np.complex64)
        result = bench.compare_outputs(candidate, reference)
        self.assertFalse(result["passed"], result)

    def test_report_records_the_magnitude_the_tolerance_was_scaled_by(self):
        reference = _reference()
        result = bench.compare_outputs(reference.copy(), reference)
        self.assertIn("reference_max_abs", result)
        self.assertAlmostEqual(result["reference_max_abs"], float(np.abs(reference).max()), places=5)
        self.assertIn("atol_effective", result)
        self.assertGreater(result["atol_effective"], bench.ATOL)

    def test_tiny_outputs_keep_the_absolute_floor(self):
        # near-zero reference: the floor must not collapse to zero and reject exact copies
        reference = (1e-9 * _reference(scale=1.0)).astype(np.complex64)
        result = bench.compare_outputs(reference.copy(), reference)
        self.assertTrue(result["passed"], result)
        self.assertGreaterEqual(result["atol_effective"], bench.ATOL)
