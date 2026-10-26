from unittest.mock import patch

import h5py
import numpy as np

from ultratrace_ulm import beamform_mach as bm, cli
from ultratrace_ulm.h5_io import acq_keys, select_acquisitions
from ultratrace_ulm.runtime import dump_pickle, load_pickle
from tests.helpers import FileTestCase, fake_beamform, write_neutral


class CliTests(FileTestCase):
    def setUp(self):
        super().setUp()
        self.raw = self.root / "raw.h5"
        write_neutral(self.raw)
        self.enterContext(patch.object(bm, "beamform_iq", side_effect=fake_beamform))
        self.enterContext(patch.object(cli, "write_track_viewer"))

    def test_run_applies_raw_selection_once(self):
        cases = [(["--acq-start", "2", "--num-acqs", "2"], [2, 3]),
                 (["--acq-step", "2", "--num-acqs", "3"], [0, 2, 4])]
        for index, (flags, expected) in enumerate(cases):
            with self.subTest(flags=flags):
                def check_selection(opts):
                    with h5py.File(opts.beamformed_path) as h5:
                        ids = select_acquisitions(acq_keys(h5), opts.acq_start, opts.num_acqs, opts.acq_step)
                        values = [int(h5[f"acquisitions/{i}/meta/compound_image"][0, 0, 0, 0].real)
                                  for i in ids]
                    self.assertEqual(values, expected)
                    return opts.tracks_path
                with patch.object(cli, "run_tracking_outputs", side_effect=check_selection):
                    cli.main(["run", "--input", str(self.raw), "--work-dir", str(self.root / str(index)), *flags])

    def test_run_preserves_selection_for_explicit_beamformed_input(self):
        with patch.object(cli, "run_tracking_outputs", return_value=self.root / "tracks.pkl") as tracker:
            cli.main(["run", "--beamformed", str(self.raw), "--work-dir", str(self.root / "explicit"),
                      "--acq-start", "2", "--num-acqs", "2", "--acq-step", "3"])
        opts = tracker.call_args.args[0]
        self.assertEqual((opts.acq_start, opts.num_acqs, opts.acq_step), (2, 2, 3))

    def test_run_calls_resumable_downloader_for_partial_input(self):
        (self.root / cli.SAMPLE_FILENAME).write_bytes(b"partial")
        with patch.object(cli, "download_sample") as downloader, patch.object(cli, "beamform_mach"), \
             patch.object(cli, "run_tracking_outputs", return_value=self.root / "tracks.pkl"):
            cli.main(["run", "--work-dir", str(self.root)])
        downloader.assert_called_once()
        self.assertFalse(downloader.call_args.kwargs["force"])

    def test_run_resumes_existing_beamformed_stage(self):
        output = self.root / "beamformed.h5"
        output.write_bytes(b"partial output")
        with patch.object(cli, "beamform_mach") as beamformer, \
             patch.object(cli, "run_tracking_outputs", return_value=self.root / "tracks.pkl"):
            cli.main(["run", "--input", str(self.raw), "--work-dir", str(self.root)])
        beamformer.assert_called_once()
        self.assertTrue(beamformer.call_args.args[0].resume)

    def test_export_default_cli_writes_smoothed_pickle_and_binary(self):
        path = self.root / "tracks.pkl"
        dump_pickle({"tracks": [{"positions": np.arange(180).reshape(60, 3),
                                  "frames": np.arange(60), "length": 60}],
                     "params": {"frame_rate_hz": 222.}}, path)
        cli.main(["export", "--tracks", str(path), "--export-dir", str(self.root / "export")])
        self.assertEqual(len(load_pickle(self.root / "tracks_smoothed.pkl")["tracks_smoothed"]), 1)
        for cutoff in (5, 20, 50):
            self.assertGreater((self.root / "export" / f"tracks_smoothed_min{cutoff}.bin").stat().st_size, 64)

    def test_export_accepts_smoothing_options(self):
        args = cli.build_parser().parse_args([
            "export", "--tracks", "tracks.pkl", "--export-dir", "out",
            "--smooth-method", "3dulm", "--smooth-window", "7", "--smooth-interp-factor", ".2",
        ])
        self.assertEqual((args.smooth_method, args.smooth_window, args.smooth_interp_factor), ("3dulm", 7, .2))
