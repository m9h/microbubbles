from dataclasses import replace
from unittest.mock import patch

import h5py
import numpy as np

from ultratrace_ulm import beamform_mach as bm
from tests.helpers import FileTestCase, fake_beamform, write_neutral


class RecoveryTests(FileTestCase):
    def setUp(self):
        super().setUp()
        self.raw = self.root / "raw.h5"
        self.output = self.root / "beamformed.h5"
        write_neutral(self.raw, range(3))
        self.opts = bm.BeamformOptions(self.raw, self.output, all_acqs=True)
        self.kernel = self.enterContext(patch.object(bm, "beamform_iq", side_effect=fake_beamform))

    def test_resume_keeps_finished_acquisitions_and_rewrites_partial_one(self):
        original_write = bm._write_acq

        def interrupt(out, out_id, compound, grid):
            if out_id == 1:
                out.require_group("acquisitions/1/meta")["compound_image"] = compound
                raise RuntimeError("simulated interruption")
            original_write(out, out_id, compound, grid)

        with patch.object(bm, "_write_acq", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                bm.beamform_mach(self.opts)
        with h5py.File(self.output, "r+") as h5:
            h5["acquisitions/0"].attrs["sentinel"] = "preserve me"
        self.kernel.reset_mock()
        bm.beamform_mach(replace(self.opts, resume=True))
        self.assertEqual([int(c.args[0].real.flat[0]) for c in self.kernel.call_args_list], [1, 2])
        with h5py.File(self.output) as h5:
            self.assertEqual(h5["acquisitions/0"].attrs["sentinel"], "preserve me")
            for aid in range(3):
                np.testing.assert_array_equal(h5[f"acquisitions/{aid}/meta/compound_image"][()], aid)
                self.assertIn("grid", h5[f"acquisitions/{aid}/meta"])

    def test_resume_of_complete_output_does_not_rebeamform(self):
        bm.beamform_mach(self.opts)
        self.kernel.reset_mock()
        bm.beamform_mach(replace(self.opts, resume=True))
        self.kernel.assert_not_called()

    def test_resume_rejects_changed_configuration_without_writing(self):
        bm.beamform_mach(self.opts)
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "(?i)(configuration|settings|match)"):
            bm.beamform_mach(replace(self.opts, resume=True, x_coarseness=2.))
        self.assertEqual(self.output.read_bytes(), before)

    def test_resume_rejects_legacy_output_without_writing(self):
        with h5py.File(self.output, "w") as h5:
            h5["precious_result"] = np.arange(4)
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "(?i)(resume|provenance|metadata)"):
            bm.beamform_mach(replace(self.opts, resume=True))
        self.assertEqual(self.output.read_bytes(), before)
