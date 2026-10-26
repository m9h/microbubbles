from dataclasses import replace

from ultratrace_ulm.runtime import load_pickle
from ultratrace_ulm.tracking import TrackingOptions, run_tracking
from tests.helpers import FileTestCase, moving_bubble, write_beamformed


class BoundaryTests(FileTestCase):
    def test_skipped_acquisitions_cannot_form_one_long_track(self):
        path = self.root / "input.h5"
        write_beamformed(path, [(0, moving_bubble(5)), (1, moving_bubble(5)), (2, moving_bubble(5, 5))])
        opts = TrackingOptions(path, self.root / "tracks.pkl", acq_step=2,
                               filter_method="none", smoothing_sigma=0, min_distance=1,
                               min_track_length=8)
        self.assertEqual(load_pickle(run_tracking(opts))["tracks"], [])

    def test_acquisition_tracks_keep_global_frames_and_source_identity(self):
        path = self.root / "input.h5"
        write_beamformed(path, [(3, moving_bubble(5)), (9, moving_bubble(7, 5))])
        opts = TrackingOptions(path, self.root / "tracks.pkl", filter_method="none",
                               smoothing_sigma=0, min_distance=1, min_track_length=4)
        data = load_pickle(run_tracking(opts))
        self.assertEqual([t["frames"].tolist() for t in data["tracks"]], [list(range(5)), list(range(5, 12))])
        self.assertEqual([t["acq_id"] for t in data["tracks"]], [3, 9])
        self.assertEqual(data["detections"]["frame_indices"].tolist(), list(range(12)))
