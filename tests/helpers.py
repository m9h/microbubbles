import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from ultratrace_ulm.beamform_core import Grid
from ultratrace_ulm.beamform_mach import _write_acq


CONFIG = dict(
    sampling_rate_hz=20e6, tx_freq_hz=5e6, speed_of_sound_m_s=1540.,
    start_depth_m=.005, end_depth_m=.010, rx_offset_s=0.,
    element_pitch_x_m=.000208, element_pitch_y_m=.001664,
    ana_mux=False, num_elev_planes=3, row_index=-1,
    mean_subtract_channels=False, large_fov=False,
    z_coarseness=1., x_coarseness=1., f_number=2.,
    tx_az_half_aperture=4, frame_rate_hz=222.,
)
Z, Y, X = np.meshgrid(
    np.linspace(.005, .010, 11), np.linspace(-.003, .003, 3),
    np.linspace(-.005, .005, 17), indexing="ij",
)
GRID = Grid(X, Y, Z)


def moving_bubble(n_frames=10, offset=0):
    e, z, x = np.indices((3, 11, 17))
    return np.array([
        100 * np.exp(-((e - 1)**2 + (z - 5)**2 +
                      (x - (5 + (f + offset) * .15))**2) / .6)
        for f in range(n_frames)
    ], dtype=np.complex64)


def write_neutral(path, ids=range(6)):
    with h5py.File(path, "w") as h5:
        h5.create_group("config").attrs.update(CONFIG)
        for aid in ids:
            group = h5.create_group(f"acquisitions/{aid}")
            group["iq_frames"] = np.full((2, 2, 2, 4, 8), aid, dtype=np.complex64)
            group["tx_delays"] = np.zeros((2, 4))
            group["tx_delays_elev"] = np.zeros((2, 2))


def fake_beamform(iq, *args):
    return np.full((2, 3, 11, 17), iq.real.flat[0], dtype=np.complex64), GRID


def write_beamformed(path, acquisitions):
    with h5py.File(path, "w") as h5:
        h5.attrs["frame_rate_hz"] = 222.
        for aid, compound in acquisitions:
            _write_acq(h5, aid, compound, GRID)


class FileTestCase(unittest.TestCase):
    def setUp(self):
        self.contexts = contextlib.ExitStack()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(self.contexts.close)
        self.root = Path(temporary.name)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def enterContext(self, context):
        # unittest.TestCase.enterContext was added in Python 3.11.
        return self.contexts.enter_context(context)
