import json

from ultratrace_ulm import cli
from ultratrace_ulm.runtime import load_pickle
from tests.helpers import FileTestCase, moving_bubble, write_beamformed


class PipelineTests(FileTestCase):
    def test_cpu_track_export_and_both_viewer_bundles(self):
        source = self.root / "beamformed.h5"
        tracks = self.root / "tracks.pkl"
        write_beamformed(source, [(0, moving_bubble(30))])
        cli.main([
            "track", "--beamformed", str(source), "--tracks", str(tracks),
            "--filter", "none", "--min-track-length", "5", "--smoothing-sigma", "0",
            "--min-distance", "1", "--export-dir", str(self.root), "--min-lengths", "5",
        ])
        smoothed = self.root / "tracks_smoothed.pkl"
        data = load_pickle(smoothed)
        self.assertEqual(len(data["tracks_smoothed"]), 1)
        self.assertEqual(data["params"]["frame_rate_hz"], 222.)
        cli.main(["track-viewer", "--tracks", str(smoothed), "--beamformed", str(source),
                  "--output-dir", str(self.root / "viewer"), "--min-length", "5"])
        cli.main(["volume", "--beamformed", str(source), "--tracks", str(smoothed),
                  "--output-dir", str(self.root / "volume"), "--max-frames", "10"])
        self.assertTrue((self.root / "viewer/index.html").is_file())
        self.assertGreater((self.root / "viewer/data/tracks.bin").stat().st_size, 64)
        metadata = json.loads((self.root / "volume/volume.json").read_text())
        self.assertEqual(metadata["frames"], 10)
        self.assertEqual(len(metadata["counts"]), 10)
