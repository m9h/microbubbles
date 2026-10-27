# Ultratrace ULM

A standalone microbubble **ULM** (ultrasound localization microscopy) pipeline:
beamforming, SVD clutter filtering, 3D bubble detection and localization,
Kalman track linking, and interactive browser viewers — packaged as a
self-contained Python package with no external acquisition stack.

Built by [Aleph Neuro](https://alephneuro.com).

## How it works

The pipeline has three stages, each exposed as a CLI subcommand:

1. **Beamform** a neutral demodulated-IQ ultratrace into a beamformed H5
   (`acquisitions/<id>/meta/compound_image` plus a saved imaging grid), using
   the optional `mach` GPU kernel. No vendor raw-frame decoding.
2. **Track** — temporal SVD clutter filtering → 3D detection and sub-voxel
   localization → Kalman + Hungarian track linking → smoothing.
3. **View** — export compact binary tracks for the animated 3D **track viewer**,
   or render a rotatable 3D SVD **volume viewer**.

The `run` subcommand chains all three (plus the data download) into one command.
The [development roadmap](ROADMAP.md) tracks GPU performance and memory work.

## Install

This project uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync                 # core: download + tracking + viewers (creates .venv)
uv sync --extra mach-cuda12   # + GPU MACH beamforming, CUDA 12.x host
uv sync --extra mach-cuda13   # + GPU MACH beamforming, CUDA 13.x host
```

`uv sync` installs the exact, locked versions (`uv.lock`). Run any command with
`uv run ultratrace-ulm ...`, or activate the environment with
`source .venv/bin/activate` and call `ultratrace-ulm` directly.

The core install has no GPU dependency. Beamforming (`beamform`, and the
beamform step of `run`) needs the `mach` extra and a CUDA host.

## Quick start

One command — download the sample data, beamform it, track, and build the 3D
track viewer:

```bash
uv run ultratrace-ulm run --spatial-tgc --frame-rate 222
cd outputs/viewer && python3 -m http.server 8080   # open http://localhost:8080
```

Already have a beamformed file? Skip download and beamforming:

```bash
ultratrace-ulm run --beamformed beamformed.h5 --frame-rate 222
```

> `run` downloads ~96 GB and beamforms on a CUDA GPU. To avoid both, pass
> `--beamformed`. For full control over any stage, use the dedicated
> subcommands below instead.

## Commands

Run `ultratrace-ulm <command> --help` for the complete flag list of any command.

| Command | Purpose |
| --- | --- |
| `download` | Fetch the public sample ultratrace (resumable). |
| `run` | End-to-end demo: download → beamform → track → track viewer. |
| `beamform` | MACH beamform a raw neutral ultratrace into a beamformed H5. |
| `track` | Streaming ULM tracking → smoothed tracks (+ optional bins). |
| `track-export` | `track` plus binary `.bin` export in one step. |
| `export` | Smooth and export `.bin` files from an existing track pickle. |
| `track-viewer` | Build the animated 3D track-flow viewer bundle. |
| `volume` | Build the rotatable 3D SVD volume viewer bundle. |
| `launcher` | Write a landing page linking several viewer bundles. |
| `doctor` | Check that the standalone runtime imports cleanly. |

### download

Resumable download. Uses [aria2](https://aria2.github.io/) (`aria2c`) for
parallel-connection downloads when it is installed (`brew install aria2` /
`apt install aria2` — recommended for this file size), and falls back to a
pure-stdlib sequential downloader otherwise; `--downloader` forces a specific
backend. A sanitized neutral
ultratrace — demodulated IQ plus transmit delays and a beamforming-only config,
no raw frames or device metadata, with the full 8-row elevation receive aperture preserved for true 3D reconstruction — hosted on Cloudflare R2 (~96 GB, 216
acquisitions):

```bash
ultratrace-ulm download --output sample_ultratrace.h5
```

An interrupted download resumes in place when re-run. Use `--force` to restart,
`--url` to point at a different source.

### run

```bash
ultratrace-ulm run \
  --work-dir outputs \
  --frame-rate 222 \
  --svd-method adaptive \
  --spatial-tgc \
  --min-length 35
```

Writes everything under `--work-dir` (default `outputs/`): the downloaded
sample, `beamformed.h5`, `tracks.pkl` / `tracks_smoothed.pkl`, `tracks_min*.bin`,
and the `viewer/` bundle. Downloads are checked and resumed, and beamforming
reuses completed acquisitions when the source and settings match. Tracking,
exports, and viewers are rebuilt. `--force` restarts downloading and beamforming.

- `--input RAW.h5` — beamform this raw file instead of downloading the sample.
- `--beamformed B.h5` — start from an existing beamformed file (skip download
  and beamforming).
- `--frame-rate` is required by adaptive SVD; the sample runs at 222 Hz.
- `--n-chunks N` controls GPU frame batching during beamforming (default 4).

### beamform

```bash
ultratrace-ulm beamform \
  --input sample_ultratrace.h5 \
  --output beamformed.h5 \
  --spatial-tgc
```

Beamform a slice with `--acq-start` / `--num-acqs` / `--acq-step`, or the whole
file with `--all-acqs`. Requires the `[mach]` extra and a CUDA GPU.

`--n-chunks` now splits each acquisition's useful frames into balanced GPU
batches. The default is 4; use 1 for a single batch, or a larger value to reduce
frame-dependent IQ and image buffers. Counts larger than the useful frame count
produce one frame per batch. Noise frames are removed once before batching.

```bash
ultratrace-ulm beamform --input sample_ultratrace.h5 --output beamformed.h5 \
  --all-acqs --n-chunks 8 --resume
# The end-to-end command exposes the same control:
ultratrace-ulm run --input sample_ultratrace.h5 --n-chunks 8 --frame-rate 222
```

Angles are compounded on the GPU and copied directly into the final host output
once per batch. Grid/transmit-arrival data are shared across batches. This keeps
the `(frames, elev, z, x)` complex64 output format and frame order. Increasing
the chunk count can increase kernel/transfer overhead; it does not reduce the
shared transmit-distance workspace or the full host input/output allocations.
Actual total GPU memory and timing still need measurement on a CUDA host.

Add `--resume` to continue an interrupted beamform with the same source,
acquisition selection, grid, and TGC settings. Completed acquisitions are
preserved; unfinished acquisitions are recomputed. The spatial TGC profile is
saved and reused so resumed images share the same normalization. Source identity
uses the resolved path, file size, and modification time.
The chunk count is an execution setting and may be changed when resuming; both
TGC estimation and output beamforming use it for acquisitions still to process.

Older beamformed files have no resume metadata and cannot be resumed safely.
Use `run --beamformed OLD.h5` for a known complete file, a new output path to
rebuild, or `run --force` to replace it. An existing filename alone does not mark
a stage complete.

### track

```bash
ultratrace-ulm track \
  --beamformed beamformed.h5 \
  --tracks outputs/tracks.pkl \
  --svd-method adaptive --frame-rate 222 \
  --sigma-threshold 2.0 --subpixel centroid \
  --tracking kalman --min-track-length 5
```

Writes `outputs/tracks.pkl` and a smoothed `outputs/tracks_smoothed.pkl`. Add
`--export-dir`/`--min-lengths` (or use `track-export`) to also emit `.bin` files.

Tracks are linked within each acquisition; they never bridge acquisition
boundaries or skipped acquisitions. Saved frames use cumulative offsets in the
selected sequence for viewing, and acquisition IDs retain their source mapping.
`--output-per-acq` saves intermediate pickles and combines only the selected IDs.

#### Recommended recipe

For non-stationary microbubble tracks, prefer **adaptive SVD with
`--temporal-sigma 0`**:

```bash
ultratrace-ulm track \
  --beamformed beamformed.h5 --tracks outputs/tracks.pkl \
  --svd-method adaptive --frame-rate 222 --knee-filter \
  --temporal-sigma 0 --sigma-threshold 2.0 --svd-low-cutoff 0.1 \
  --min-distance 2 --smoothing-sigma 1.0 --tracking kalman \
  --max-gap 3 --min-track-length 5 --max-cost 10
```

A large `--temporal-sigma` (e.g. 7) with the `fast` SVD variant tends to track
stationary tissue/clutter: many more tracks that barely move. Adaptive SVD with
no temporal blur matches the production reference (~260 tracks/acquisition,
genuinely flowing).

### export

Re-smooth and re-export `.bin` files from an existing pickle without re-tracking:

```bash
ultratrace-ulm export \
  --tracks outputs/tracks.pkl \
  --export-dir outputs \
  --min-lengths 5 20 50
```

### track-viewer

Build the animated 3D track-flow viewer (Three.js point-flow):

```bash
ultratrace-ulm track-viewer \
  --tracks outputs/tracks_smoothed.pkl \
  --output-dir outputs/viewer \
  --min-length 35 --beamformed beamformed.h5
cd outputs/viewer && python3 -m http.server 8080
```

`--beamformed` is optional and only used to sample per-point B-mode intensity.
Subset intensity lookup uses `params.selected_acq_ids` from the tracks pickle.
For older subset pickles without that mapping, re-track or omit `--beamformed`.

### volume

Build the rotatable 3D SVD volume viewer with track overlay:

```bash
ultratrace-ulm volume \
  --beamformed beamformed.h5 \
  --tracks outputs/tracks_smoothed.pkl \
  --output-dir outputs/volume
cd outputs/volume && python3 -m http.server 8080
```

### launcher

Serve several viewer bundles behind one landing page. Export the `track-viewer`
and `volume` bundles into a shared directory, then:

```bash
ultratrace-ulm launcher \
  --output-dir outputs \
  --title "Ultratrace ULM" \
  --subtitle "adaptive SVD" \
  --viewer "viewer/|3D Track-Flow Viewer|Animated point-flow of tracks.|✨" \
  --viewer "volume/|3D SVD Volume Viewer|Rotatable super-resolution volume.|🧊"
```

This writes `index.html` + `viewers.json`. Each `--viewer` is
`HREF|TITLE|DESCRIPTION|EMOJI` (only the href is required) and may be repeated.

## Viewing

Every viewer bundle is static — serve its directory with any file server:

```bash
cd outputs/viewer && python3 -m http.server 8080   # http://localhost:8080
```

## Tests

The CPU suite uses small synthetic H5 files and needs neither CUDA nor the
96 GB sample:

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
```

It covers acquisition boundaries, interrupted beamforming, resume compatibility,
TGC reuse, subset selection, CLI export, viewer mapping, and GPU-compounding
layout through a CPU stand-in. Chunking checks also cover frame ordering,
uneven batches, noise removal, buffer lifetimes, and CLI validation before any
download or output write. GitHub Actions runs it on Python 3.10 and 3.14.

On a CUDA host, also compare real MACH/CuPy compounding with host summation and
compare chunked execution against that reference:

```bash
uv sync --frozen --extra mach-cuda12
ULM_TEST_CUDA=1 .venv/bin/python -m unittest tests.test_beamform_compound -v
```

Use `mach-cuda13` for a CUDA 13 host. The real CUDA test is skipped unless
`ULM_TEST_CUDA=1`; CPU checks do not establish GPU performance or peak memory.

## License

[MIT](LICENSE) — © 2026 Aleph Neuro.
