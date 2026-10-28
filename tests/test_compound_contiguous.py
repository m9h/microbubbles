"""The MACH kernel must receive C-contiguous channel data.

`beamform_iq` hands the kernel a transposed view `(1, n_rx, n_samples, frames)`; mach then
warns "array is not contiguous, rearranging will add latency" and rearranges it internally
on every angle of every batch. On a GB10 (2026-10-27, 128 frames x 9 angles) per-call time
grew ~70x for a 16x larger batch, with that warning on every case. Making the array
contiguous on the device before the call removes the in-kernel rearrangement.
"""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

from ultratrace_ulm import beamform_core as core
from tests.helpers import CONFIG, GRID


class ContiguousChannelDataTests(unittest.TestCase):
    def test_kernel_receives_c_contiguous_channel_data_for_every_angle_and_batch(self):
        cp = ModuleType("cupy")
        cp.__dict__.update({k: v for k, v in np.__dict__.items() if not k.startswith("__")})
        cp.asnumpy = Mock(side_effect=lambda a, out=None: np.array(a) if out is None else (np.copyto(out, a), out)[1])
        config = core.NeutralConfig.from_h5_attrs(CONFIG)
        angles, frames, rows, cols, samples = 3, 5, 2, 4, 8
        seen = []

        def kernel(**kwargs):
            data = kwargs["channel_data"]
            seen.append((data.shape, bool(data.flags.c_contiguous)))
            return np.zeros((GRID.x.size, data.shape[-1]), dtype=np.complex64)

        iq = (np.arange(frames * angles * rows * cols * samples, dtype=np.float32)
              .reshape(frames, angles, rows, cols, samples).astype(np.complex64))
        for n_chunks in (1, 2):
            seen.clear()
            with patch.dict(sys.modules, {"cupy": cp}), \
                 patch.object(core, "MACH_AVAILABLE", True), \
                 patch.object(core, "build_grid", return_value=GRID), \
                 patch.object(core, "_mach_beamform", side_effect=kernel):
                core.beamform_iq(iq, np.zeros((angles, cols)), np.zeros((angles, rows)), config, n_chunks=n_chunks)
            with self.subTest(n_chunks=n_chunks):
                self.assertEqual(len(seen), angles * n_chunks)
                for shape, contiguous in seen:
                    self.assertEqual(shape[:2], (1, rows * cols))
                    self.assertEqual(shape[2], samples)
                    self.assertTrue(contiguous, f"non-contiguous channel_data of shape {shape} reached the kernel")
