"""Frame batching must bound device allocations without changing output frames."""

import sys
import unittest
import weakref
from dataclasses import replace
from types import ModuleType
from unittest.mock import patch

import h5py
import numpy as np

from ultratrace_ulm import beamform_core as core, beamform_mach as bm, cli
from tests.helpers import CONFIG, GRID, FileTestCase, fake_beamform, write_neutral


class FrameChunkTests(unittest.TestCase):
    def test_batches_preserve_frames_after_noise_removal_and_bound_device_buffers(self):
        config = replace(core.NeutralConfig.from_h5_attrs(CONFIG), num_noise_loops=2)
        iq = np.empty((9, 4, 2, 4, 8), dtype=np.complex64)
        for frame in range(9):
            for angle in range(4):
                iq[frame, angle] = frame * 10 + angle
        expected = np.empty((7, 3, 11, 17), dtype=np.complex64)
        points = np.arange(GRID.x.size, dtype=np.float32).reshape(GRID.x.shape).transpose(1, 0, 2)
        for frame in range(7):
            expected[frame] = 4 * points + 1j * (4 * (frame + 2) * 10 + 6)

        for count, sizes in ((1, [7]), (3, [3, 2, 2]), (20, [1] * 7)):
            with self.subTest(n_chunks=count):
                cp = ModuleType("cupy")
                cp.__dict__.update({k: v for k, v in np.__dict__.items() if not k.startswith("__")})
                uploads, accumulators, transfers = [], [], []
                last_iq = last_compound = None

                def device_array(array, **kwargs):
                    nonlocal last_iq
                    if array.ndim == 5:
                        if last_iq is not None:
                            self.assertIsNone(last_iq(), "previous IQ chunk is still live")
                            self.assertIsNone(last_compound(), "previous compound chunk is still live")
                        result = np.array(array, **kwargs)
                        uploads.append(result.shape[0])
                        last_iq = weakref.ref(result)
                        return result
                    return np.asarray(array, **kwargs)

                def zeros(shape, **kwargs):
                    nonlocal last_compound
                    result = np.zeros(shape, **kwargs)
                    accumulators.append(shape[1])
                    last_compound = weakref.ref(result)
                    return result

                def host_array(array, out=None):
                    self.assertIsNotNone(out, "copy directly into the preallocated host output")
                    self.assertTrue(out.flags.c_contiguous)
                    transfers.append(weakref.ref(out))
                    np.copyto(out, array)
                    return out

                def kernel(channel_data, **kwargs):
                    markers = channel_data[0, 0, 0, :]
                    return (np.arange(GRID.x.size)[:, None] + 1j * markers[None, :]).astype(np.complex64)

                cp.asarray, cp.zeros, cp.asnumpy = device_array, zeros, host_array
                with patch.dict(sys.modules, {"cupy": cp}), \
                     patch.object(core, "MACH_AVAILABLE", True), \
                     patch.object(core, "build_grid", return_value=GRID), \
                     patch.object(core, "_mach_beamform", new=kernel), \
                     patch.object(core, "preprocess", wraps=core.preprocess) as preprocess:
                    result, grid = core.beamform_iq(
                        iq, np.zeros((4, 4)), np.zeros((4, 2)), config, n_chunks=count,
                    )
                preprocess.assert_called_once()
                np.testing.assert_array_equal(result, expected)
                self.assertEqual(result.dtype, np.complex64)
                self.assertTrue(result.flags.c_contiguous)
                self.assertIs(grid, GRID)
                self.assertEqual(uploads, sizes)
                self.assertEqual(accumulators, sizes)
                self.assertEqual(len(transfers), len(sizes))

    def test_invalid_chunk_counts_fail_before_gpu_import(self):
        config = core.NeutralConfig.from_h5_attrs(CONFIG)
        with patch.object(core, "MACH_AVAILABLE", False):
            for count in (0, -2, 1.5):
                with self.subTest(n_chunks=count), self.assertRaisesRegex(ValueError, "positive integer"):
                    core.beamform_iq(None, None, None, config, n_chunks=count)


class FrameChunkIntegrationTests(FileTestCase):
    def setUp(self):
        super().setUp()
        self.raw = self.root / "raw.h5"
        self.output = self.root / "beamformed.h5"
        write_neutral(self.raw, range(2))
        self.kernel = self.enterContext(patch.object(bm, "beamform_iq", side_effect=fake_beamform))

    def test_chunk_count_reaches_both_tgc_and_output_beamforming(self):
        opts = bm.BeamformOptions(self.raw, self.output, n_chunks=3, spatial_tgc=True, tgc_acqs=1)
        with patch.object(bm, "compute_global_tgc", return_value=np.ones((1, 3, 11, 17))):
            bm.beamform_mach(opts)
        self.assertEqual(len(self.kernel.call_args_list), 2)
        self.assertEqual([call.kwargs.get("n_chunks") for call in self.kernel.call_args_list], [3, 3])
        self.kernel.reset_mock()
        # Chunk count is an execution choice; changing it may resume the same output.
        bm.beamform_mach(replace(opts, n_chunks=8, resume=True))
        self.kernel.assert_not_called()

    def test_invalid_chunk_count_cannot_truncate_output(self):
        with h5py.File(self.output, "w") as h5:
            h5["precious"] = [1, 2, 3]
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            bm.beamform_mach(bm.BeamformOptions(self.raw, self.output, n_chunks=0))
        self.assertEqual(self.output.read_bytes(), before)
        self.kernel.assert_not_called()

    def test_run_exposes_chunk_control(self):
        with patch.object(cli, "beamform_mach") as beamformer, \
             patch.object(cli, "run_tracking_outputs", return_value=self.root / "tracks.pkl"), \
             patch.object(cli, "write_track_viewer"):
            cli.main(["run", "--input", str(self.raw), "--work-dir", str(self.root), "--n-chunks", "8"])
        self.assertEqual(beamformer.call_args.args[0].n_chunks, 8)

    def test_invalid_run_chunk_count_fails_before_any_stage(self):
        with patch.object(cli, "download_sample") as downloader, \
             patch.object(cli, "beamform_mach") as beamformer, \
             patch.object(cli, "run_tracking_outputs", return_value=self.root / "tracks.pkl"), \
             patch.object(cli, "write_track_viewer"):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["run", "--work-dir", str(self.root), "--n-chunks", "0"])
        self.assertEqual(raised.exception.code, 2)
        downloader.assert_not_called()
        beamformer.assert_not_called()
