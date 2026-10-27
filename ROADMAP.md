# GPU performance and memory roadmap

The current development priority is GPU performance and memory scaling. This
document records the next milestones and the evidence needed to complete them.

## Foundation — merged

- Compound angles on the GPU instead of stacking full angle volumes on the host.
- Resume compatible acquisitions without overwriting completed results.
- Keep acquisition identities and tracking boundaries intact.
- Run CPU regressions in CI, with an opt-in real CUDA comparison.

## 1. GPU frame batching — implemented; CUDA validation pending

Make the existing `--n-chunks` control functional in `beamform` and `run`.
Preprocess once, share the grid/transmit arrivals, and hold one frame batch's IQ
and compound buffers on the GPU. Copy each result into the final host array.
Apply the same setting during spatial TGC estimation.

CPU regressions verify exact composition, useful-frame ordering, uneven batches,
buffer lifetimes, and rejection of invalid settings before writes/downloads.
The opt-in CUDA test compares host compounding with both single-batch and
chunked MACH execution. Run it on supported CUDA hardware before declaring
hardware validation complete; no GPU timing or peak-memory result is recorded yet.

## 2. Reproducible CUDA performance measurements — next

- Add a small benchmark fixture and a command that records hardware, CUDA/CuPy/
  MACH versions, grid dimensions, frame/angle counts, and chunk settings.
- Report warm-up separately from repeated timings and compare chunked output
  against the same numerical reference.
- Measure device allocations and host memory; distinguish live allocation peaks
  from memory retained by CuPy's pool. Compare settings in isolated processes so
  an earlier large allocation cannot distort later results.
- Save machine-readable results and a documented repeatable command. Use these
  measurements to choose batch defaults and assess the next optimizations.

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
