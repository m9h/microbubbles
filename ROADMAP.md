# GPU performance and memory roadmap

The current development priority is GPU performance and memory scaling. This
document records the next milestones and the evidence needed to complete them.

## Foundation — merged

- Compound angles on the GPU instead of stacking full angle volumes on the host.
- Resume compatible acquisitions without overwriting completed results.
- Keep acquisition identities and tracking boundaries intact.
- Run CPU regressions in CI, with an opt-in real CUDA comparison.

## 1. GPU frame batching — merged; GB10 CUDA validation reported

Make the existing `--n-chunks` control functional in `beamform` and `run`.
Preprocess once, share the grid/transmit arrivals, and hold one frame batch's IQ
and compound buffers on the GPU. Copy each result into the final host array.
Apply the same setting during spatial TGC estimation.

CPU regressions verify exact composition, useful-frame ordering, uneven batches,
buffer lifetimes, and rejection of invalid settings before writes/downloads.
The opt-in CUDA test compares host compounding with both single-batch and
chunked MACH execution. The [PR #2 review](https://github.com/m9h/microbubbles/pull/2#issuecomment-6454827137)
reports eight CUDA tests passing on a DGX Spark/GB10 with CuPy 14.2.0 and a
locally built MACH 0.2.1. This validates that reported environment, not every
supported device or the exact locked dependencies. PR #2 merged on October 27.

## 2. Reproducible CUDA performance measurements — GB10 memory results reported; timing validation pending

The [benchmark command and protocol](BENCHMARKS.md) now provide a deterministic
synthetic fixture, separate worker processes for each setting, synchronized
warm-up/repeated timings, a host-compounded numerical reference, and JSON output.
An additional instrumented pass distinguishes CuPy pool live/reserved peaks;
process RSS is recorded before output serialization. CPU protocol tests and an
opt-in CUDA smoke test cover the harness. PR #3 merged on October 27.

[PR #4](https://github.com/m9h/microbubbles/pull/4) adds a magnitude-scaled
comparison floor, contiguous kernel input and early device initialization for
GB10. It reports 21 passing CUDA-enabled tests and a reduction in CuPy live peak
from 1417 MiB at one chunk to 272 MiB at eight/sixteen chunks for the larger
fixture. See [BENCHMARKS.md](BENCHMARKS.md#reported-gb10-results) for the reported
table and limits. The development Mac has not independently rerun CUDA tests.
Timings vary substantially; these results do not establish a speedup.

Remaining acceptance work:

- Archive the original GB10 JSON reports with exact hardware/software provenance;
  only the PR's summary table is currently available in the repository.
- Collect stable small/larger-fixture timings across frame/chunk counts under
  controlled conditions or on another CUDA host, and run the CUDA checks there.
- Review numerical errors, synchronized timings, host RSS, and CuPy allocation
  peaks; profile allocations outside CuPy separately before claiming total
  device-memory savings.
- Use reproducible measurements to choose batch defaults and assess the next
  optimizations; keep the existing defaults until then.

## 3. Bound remaining workspaces and host memory — planned

- Make transmit-distance workspace sizing configurable or memory-budgeted;
  frame batching does not shrink this allocation.
- Read neutral IQ in frame slices and stream output slices into H5, keeping
  acquisition completion markers and resume compatibility correct.
- Accumulate the global TGC power map acquisition by acquisition instead of
  retaining every calibration compound. Preserve the temporal SVD calculation.

Acceptance requires chunked/streamed equivalence checks, interrupted-write tests,
and measured memory scaling with acquisition/frame counts.

## 4. Reduce kernel and transfer overhead — measurement-driven

Evaluate MACH's existing output-buffer accumulation and reuse of GPU geometry
across acquisitions. Retain changes only when benchmarks show a benefit and
complex-IQ, SVD-filtered magnitude, and tracking comparisons remain within
documented tolerances. Record changes in floating-point accumulation explicitly.

All implementation changes follow red–green TDD. CPU stand-ins test composition
and control flow; real CUDA correctness and performance require CUDA execution.
