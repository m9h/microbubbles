# Development handoff — reviewed 2026-10-28 (America/Los_Angeles)

Originally saved before a system reboot; updated after resuming development.
Read this file, `ROADMAP.md`, and `BENCHMARKS.md` before continuing.

## Repository and current state

- GitHub: https://github.com/m9h/microbubbles (plural).
- Local checkout: `/Users/mhough/Workspace/microbubbles`.
- Latest review branch: `bench-tolerance-contiguous` for
  [PR #4](https://github.com/m9h/microbubbles/pull/4), implementation `95f8b0f`.
- PRs #2 and #3 merged into `main` on October 27; main was `126f23a` before
  PR #4. Check the live PR state for the outcome of this review/merge session.
- Foundation fixes are merged into `main` at `e5b9a14` via
  [PR #1](https://github.com/m9h/microbubbles/pull/1).
- Frame batching is merged in [PR #2](https://github.com/m9h/microbubbles/pull/2);
  the benchmark harness is merged in [PR #3](https://github.com/m9h/microbubbles/pull/3).
- GitHub Actions passed on Python 3.10 and 3.14 for `95f8b0f`.
- No sample dataset has been downloaded. The only manual benchmark invocation
  on this Mac produced an expected missing-CUDA failure report in `/private/tmp`.

## User direction

The user requested review of the project and `gpu-compound`, fixes and merging,
then continued roadmap development. They explicitly require **red–green TDD**
and selected **GPU performance and memory scaling** as the development priority.
GitHub issues are disabled; `ROADMAP.md` records the milestones.

The latest technical question compared the Kidger SciML ecosystem with SCICO.
The response was an architecture recommendation, not an accepted migration plan.
No JAX, Equinox, Optimistix, Diffrax, Lineax, or SCICO migration has been started.
The latest user request is to check the other agent's updates and merge the new
branch. No numerical backend migration has been started.

## Completed foundation (PR #1, merged)

- GPU angle compounding without retaining all angle volumes on the host.
- Safe beamforming resume: source/configuration compatibility, acquisition
  completion markers, partial-acquisition rebuild, and saved TGC normalization.
- Tracking stays within acquisition boundaries, with cumulative viewer frame
  offsets and preserved source acquisition IDs.
- Acquisition selection is applied once in the end-to-end pipeline.
- Export smoothing CLI options are defined.
- Viewer intensity sampling uses the saved selected acquisition IDs.
- End-to-end execution resumes downloads and beamforming instead of treating
  an existing partial file as a completed stage.
- Cached per-acquisition track combination respects selection and frame offsets.
- CPU regression CI and an opt-in real CUDA comparison.

## Completed frame batching (PR #2, merged)

- Red tests: `e7bac89`; implementation and roadmap: `f2f2a9e`.
- `beamform_iq(..., n_chunks=...)` validates positive integers, preprocesses IQ
  once, removes noise frames once, and shares geometry/transmit arrivals.
- Useful frames are split into balanced, nonempty batches. Each batch compounds
  all angles on the GPU and copies directly into its final host output slice.
- Batch device buffers are released before the next upload. Chunk counts larger
  than the useful frame count produce one frame per batch.
- Both TGC estimation and regular beamforming receive the chunk setting.
- `run --n-chunks` is exposed; invalid counts fail before downloads or writes.
- Chunk count is deliberately excluded from the resume compatibility signature
  because it changes execution, not the requested output.
- Full host IQ/output arrays and the transmit-distance workspace still exist;
  frame batching does not solve those memory costs.

Validation recorded at the original frame-batching checkpoint: 26 tests total,
25 passing and 1 real-CUDA test skipped. CPU stand-ins cover exact composition,
frame order, uneven/excessive chunk counts, noise removal, and device-buffer
lifetimes. No real CUDA correctness run, timing, or peak-memory measurement has
been performed in this local session (Darwin arm64 without CUDA).

## CUDA benchmark harness — merged; GB10 results reported

- Red tests committed at `d3c4fd2` before implementing the benchmark module.
- Command: `.venv/bin/python -m ultratrace_ulm.benchmark --output outputs/cuda-benchmark.json`.
- Deterministic configurable synthetic IQ, full fixture/config/software/device
  provenance, isolated subprocesses, synchronized warm-up and repeat timings.
- A separate memory pass records CuPy pool live/reserved peaks without adding
  instrumentation to timed calls. Host RSS is a process-lifetime high-water mark.
- Host angle compounding uses single-frame device batches so the reference can
  fit even if an unchunked measured case runs out of GPU memory. A regression
  failed before changing the reference to single-frame batches, then passed.
- Every setting compares its memory-pass complex64 output against the common
  reference. Failed cases are reported and later settings still run.
- Original harness local suite: 37 tests, 35 passing and 2 opt-in CUDA tests skipped.
  Runtime doctor, CLI help, missing-CUDA failure reporting and whitespace checks
  passed at that checkpoint. See the updated hardware report below.
- Read `BENCHMARKS.md` before interpreting metrics: CuPy counters exclude other
  allocators and CUDA context memory, and do not establish total VRAM savings.

## GB10 follow-up and PR #4 review

- Claude Opus 5.5 co-authored `95f8b0f`, adding a magnitude-scaled benchmark
  tolerance, contiguous channel input to MACH, and a first device allocation
  before constructing large host fixtures.
- The PR #2 review reports eight passing CUDA tests on GB10, using CuPy 14.2.0
  and locally built MACH 0.2.1 on aarch64. PR #4 reports 21 CUDA-enabled tests
  passing and a larger-fixture CuPy live peak of 1417 MiB at one chunk versus
  272 MiB at eight/sixteen chunks.
- These are the other agent's reported hardware results. No independent CUDA
  rerun was possible on this Mac. Original machine-readable reports still need
  archiving; BENCHMARKS.md links the source and reproduces its summary table.
- PR #4 local review: 44 tests run, 42 passed and 2 CUDA opt-ins skipped;
  runtime doctor and whitespace checks passed. Review also reconciles benchmark
  tolerance documentation and stale roadmap/handoff status.
- Timing varies widely (2.2–33.6 seconds within one setting). Keep the existing
  defaults and make no speedup claim; collect stable timings before choosing
  further kernel optimizations.

## Architecture recommendation discussed

Keep microbubbles independent, prototype a selective JAX backend, and introduce
Kidger libraries only where they solve a concrete problem. SCICO can be an
optional reconstruction layer. Neither stack alone establishes a speed or
memory improvement over the existing NumPy/SciPy and MACH/CuPy implementation.

| Component | Suggested role |
| --- | --- |
| Beamforming | Retain MACH initially; benchmark any replacement. |
| Temporal SVD/clutter filtering | Plain JAX linear algebra with bounded workspaces. |
| Parameterized PSF/calibration models | Equinox, if useful. |
| Batched subpixel PSF fitting | Optimistix nonlinear least squares. |
| Linear solves | Lineax where needed; it does not replace SVD/eigendecomposition. |
| Sparse deconvolution/model-based reconstruction | SCICO operators, regularizers, and proximal solvers. |
| Future continuous flow or bubble dynamics | Diffrax ODE/SDE solvers. |

The current project processes measured IQ through beamforming, SVD, detection,
and discrete Kalman/Hungarian tracking. It does not currently solve a bubble
physics ODE or optimize an imaging inverse-problem objective. SCICO becomes a
stronger foundation if the scientific objective changes to recovering overlapping
bubbles using a calibrated PSF or reconstructing from channel measurements.
That requires a validated forward model and adjoint; wrapping MACH does not
automatically provide these or autodifferentiation.

Recommended next architecture experiment: spatially tiled temporal SVD.
Keep all temporal frames while accumulating the small temporal Gram matrix over
voxel blocks, then filter blocks with the shared temporal basis. Preserve the
existing adaptive cutoff semantics: centered data for cutoff estimation,
uncentered data for the subsequent projection. Compare NumPy, CuPy, and JAX
with the same fixtures and memory budget before selecting a backend.

Migration constraints to retain:

- Existing eigendecomposition/SVD work uses `complex128`. JAX X64 is disabled
  by default; explicitly preserve precision and test cutoff/magnitude/track
  behavior. Double precision does not guarantee bitwise equality across backends.
- JAX normally preallocates 75% of GPU memory. Coordinate its allocator with
  MACH/CuPy when sharing a device; measure live and reserved memory separately.
- Variable-length detections and discrete Hungarian assignment do not become
  differentiable simply by moving arrays to JAX. Retain CPU handling or design
  padded/masked representations where appropriate.
- Keep H5 IO, resume/checkpoint logic, CLI, and viewers independent of numerical
  backend selection.

Official references consulted:

- https://docs.kidger.site/equinox/
- https://docs.kidger.site/optimistix/
- https://docs.kidger.site/lineax/
- https://docs.kidger.site/diffrax/
- https://scico.readthedocs.io/en/stable/inverse.html
- https://docs.jax.dev/en/latest/101/default_dtypes.html
- https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- https://docs.jax.dev/en/latest/ffi.html
- https://docs.jax.dev/en/latest/_autosummary/jax.dlpack.from_dlpack.html

## Resume after reboot

```bash
cd /Users/mhough/Workspace/microbubbles
git status --short --branch
git fetch origin
gh pr view 4 --repo m9h/microbubbles
```

Read `ROADMAP.md` for the committed sequence. Its next step is archiving the GB10
reports and collecting stable CUDA timings; the tiled-SVD/JAX experiment above is the later recommendation
from the architecture discussion, not a completed change to that sequence.

For CPU verification when resuming implementation:

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ultratrace-ulm doctor
```

On a supported CUDA machine, validate the real beamformer:

```bash
uv sync --frozen --extra mach-cuda12
ULM_TEST_CUDA=1 .venv/bin/python -m unittest tests.test_beamform_compound tests.test_benchmark -v
```

Use `mach-cuda13` instead for a CUDA 13 host. Do not count CPU stand-ins as CUDA
validation. Benchmarks should separate warm-up/compilation from repeated timing,
record device and package versions and fixture shapes, compare numerical output,
and measure host memory plus device live/pool peaks in isolated processes.

For a tiled-SVD prototype, first write failing reference-equivalence and memory-
bound tests, then implement the smallest change that passes. Check adaptive
cutoffs, complex filtered output, magnitude, and downstream detections/tracks
with documented tolerances. Keep the current implementation as a reference.

Other planned optimizations remain: bound transmit-distance workspace, stream
H5 frame IO, accumulate calibration TGC maps without retaining all compounds,
and benchmark MACH output-buffer accumulation and geometry reuse.
