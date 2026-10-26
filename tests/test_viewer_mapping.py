import numpy as np

from ultratrace_ulm.runtime import dump_pickle
from ultratrace_ulm.track_viewer_export import export_tracks_bin_v3
from tests.helpers import FileTestCase, X, Y, Z, write_beamformed


class ViewerMappingTests(FileTestCase):
    def test_subset_intensities_follow_saved_acquisition_ids(self):
        h5_path = self.root / "input.h5"
        acquisitions = []
        for aid in range(3):
            volume = np.ones((2, 3, 11, 17), dtype=np.complex64)
            volume[:, 1, 5, 5:8] = [10, 20, 30] if aid == 0 else [30, 20, 10]
            acquisitions.append((aid, volume))
        write_beamformed(h5_path, acquisitions)
        positions = np.array([[X[5, 1, j], Y[5, 1, j], Z[5, 1, j]] for j in (5, 6, 7)]) * 1000
        pkl = self.root / "tracks.pkl"
        dump_pickle({"tracks": [dict(positions=positions, frames=np.zeros(3), length=3)],
                     "n_acquisitions": 1, "params": {"selected_acq_ids": [0]},
                     **{f"grid_{name}": array.transpose(1, 0, 2) * 1000
                        for name, array in zip("xyz", (X, Y, Z))}}, pkl)
        output = self.root / "tracks.bin"
        export_tracks_bin_v3(pkl, output, min_length=1, beamformed=h5_path)
        points = np.frombuffer(output.read_bytes(), dtype="<f4", offset=64 + 8).reshape(3, 6)
        np.testing.assert_allclose(points[:, 5], [0, .5, 1], atol=1e-5)
