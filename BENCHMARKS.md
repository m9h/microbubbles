# CUDA beamforming benchmarks

The harness has reported CUDA correctness and memory results on a DGX Spark
(GB10); stable timing and archived JSON reports remain outstanding. See the
[hardware results below](#reported-gb10-results). CPU tests check the measurement
protocol, not CUDA correctness, throughput, or total GPU memory.

## Run on a CUDA host

From the repository root, install the extra matching the host and run both real
CUDA checks before collecting measurements:

```bash
uv sync --frozen --extra mach-cuda12
ULM_TEST_CUDA=1 .venv/bin/python -m unittest tests.test_beamform_compound tests.test_benchmark -v
.venv/bin/python -m ultratrace_ulm.benchmark \
  --output outputs/cuda-benchmark-small.json \
  --frames 16 --angles 4 --rows 2 --cols 4 --samples 512 --elev-planes 3 \
  --chunks 1 2 4 8 --warmups 1 --repeats 5 --seed 3
```

Use `mach-cuda13` for a CUDA 13 host. The installed environment's Python is used
for every child process. The [GB10 validation report](https://github.com/m9h/microbubbles/pull/2#issuecomment-6454827137)
used a local MACH 0.2.1 source build on aarch64 because the available PyPI wheel
was x86_64-only; the extra alone did not supply MACH on that machine. That report
used CuPy 14.2.0, so it is not validation of the exact locked dependency set.
The command generates deterministic complex64 random
IQ and zero transmit delays locally; it does not download the 96 GB sample.
It is a numerical/performance fixture, not a physical microbubble simulation.

Increase frames and aperture to study scaling, for example:

```bash
.venv/bin/python -m ultratrace_ulm.benchmark \
  --output outputs/cuda-benchmark-larger.json \
  --frames 128 --angles 9 --rows 8 --cols 32 --samples 512 --elev-planes 25 \
  --chunks 1 2 4 8 16 --warmups 2 --repeats 7 --seed 3
```

Compare multiple frame counts on the same otherwise idle device; record each
run in a separate file. The synthetic aperture sets the azimuthal grid width,
and `--elev-planes` sets its height. The full acquisition config, actual grid
dimensions, input hash, shapes, dtypes, seed, package versions, GPU model and
capacity, compute capability, CUDA runtime/driver versions, selected environment
settings, git revision, and dirty flag are recorded in the JSON report.

## Measurement protocol

Each reference and each chunk setting runs sequentially in a fresh interpreter.
No setting inherits another setting's CUDA context or memory pool. CuPy's disk
kernel cache can be reused across processes: warm-up is **not** a guaranteed
cold compilation measurement. Each worker performs a small device allocation
and synchronizes before constructing its host fixture. This initialization is
outside timed calls and addresses the intermittent first-allocation failure
reported on GB10 after large NumPy allocations.

1. Build a reference in its own process by copying each real MACH angle result
   to the host and summing in complex64. Use single-frame GPU batches so the
   reference does not require the largest measured batch to fit. Validate that
   the reference is finite, nonzero, and agrees with GPU compounding.
2. For each setting, construct the same fixture and record warm-up wall times
   separately. Synchronize the device immediately before and after every timed
   call. Repeated measurements include preprocessing, geometry, GPU transfers,
   beamforming, and host output construction inside `beamform_iq`; they exclude
   fixture creation, process startup, output serialization, and comparison.
3. Keep the pool warm between timed repetitions. Drop each returned image and
   grid before starting the next call. Report every repeat plus its median.
4. After timing, synchronize and free idle CuPy pool blocks. Run one additional
   memory-instrumented call with warmed kernels. Observe the pool immediately
   after every allocation, including reuse. This instrumentation is excluded
   from the timing samples.
5. Save that call's output to a temporary file, then compare it with the common
   reference in the coordinator. Require matching shape, complex64, finite
   values, a nonzero reference, and elementwise
   `abs(actual - reference) <= atol_effective + 2e-6 * abs(reference)`, with
   `atol_effective = 1e-6 + 1e-5 * max(abs(reference))` across the whole reference.
   Record `reference_max_abs`, `atol_effective`, maximum absolute error and
   relative L2 error. The magnitude-scaled floor accommodates the discrepancies
   reported for the larger GB10 fixture. It is more permissive for weak voxels
   than a purely local relative tolerance and is specific to this benchmark.
   The separate small CUDA composition test retains its original tolerances.

This checks compounding, batching and layout against the same MACH kernel. It
does not independently validate the beamforming physics, SVD cutoffs, detections,
or tracks. Validation uses the separate memory pass, not every timed repetition.

## Memory fields and limits

| JSON field under each case's `memory` | Meaning |
| --- | --- |
| `cupy_pool_live_peak_bytes` | Maximum bytes in use in the observed CuPy pool during the extra memory pass. Includes allocation rounding. |
| `cupy_pool_reserved_peak_bytes` | Maximum total pool bytes, including unused cached blocks, during that pass. |
| `cupy_pool_live_after_bytes` | Pool bytes still in use after the call returns and synchronizes. |
| `cupy_pool_reserved_after_bytes` | Pool bytes retained after that call. |
| `process_peak_rss_bytes` | Worker process RSS high-water mark through measurement, including imports, fixture construction, warm-up and repetitions, before saving output. `null` where `resource` is unavailable. |

The device fields use CuPy's [allocator context](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.using_allocator.html)
and [pool counters](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPool.html).
They cover allocations routed through that pool in the beamforming thread.
CUDA contexts, direct kernel/library allocations, other allocators (including
PyTorch), and other processes are **not** included. These are not total VRAM
peaks; use an external device profiler when total device memory is needed.
Reserved minus live peaks is not a simultaneous cached-memory measurement.

Host RSS is an absolute process-lifetime high-water mark, not a per-call delta.
The reference worker's host angle copies and coordinator comparison are excluded
from measured workers. Peak RSS includes fixture-generation temporaries and the
full host IQ/output arrays. A larger chunk count does not bound those arrays or
the transmit-distance workspace.

If a setting fails (including out-of-memory), its error is saved and later
settings still run in fresh processes. A failed reference aborts the comparisons.
The CLI writes a JSON failure report and exits nonzero on missing CUDA, worker
failure, or numerical mismatch. Invalid arguments fail before any worker or file
creation. Reports are replaced atomically; temporary output arrays are removed.

CPU protocol checks:

```bash
.venv/bin/python -m unittest tests.test_benchmark -v
```

## Reported GB10 results

[PR #4](https://github.com/m9h/microbubbles/pull/4), co-authored by Claude Opus 5.5,
reports the following results for the 128-frame, 9-angle larger fixture on an
otherwise idle DGX Spark. These values are transcribed from the PR summary;
raw JSON reports have not been committed or attached there, and this Mac review
did not independently rerun CUDA measurements.

| Chunks | Comparison | Relative L2 error | CuPy live peak | Process RSS | Median time |
| --- | --- | --- | --- | --- | --- |
| 1 | Passed | 1.2e-7 | 1417 MiB | 4.8 GB | 8.6 s |
| 2 | Passed | 1.2e-7 | 710 MiB | 3.8 GB | 9.6 s |
| 4 | Passed | 1.2e-7 | 357 MiB | 3.3 GB | 4.7 s |
| 8 | Passed | 1.2e-7 | 272 MiB | 3.0 GB | 5.6 s |
| 16 | Passed | 1.2e-7 | 272 MiB | 2.9 GB | 6.7 s |

The PR reports 21 passing tests with CUDA opt-ins enabled. Its memory results
support reduced frame-dependent allocations and a remaining workspace floor;
the numbers describe the CuPy pool, not total device memory. Individual timing
samples ranged from 2.2 to 33.6 seconds for one setting, so these medians do not
establish a speedup or justify changing the default chunk count. Unified-memory
migration or clock behavior were suggested explanations, not established causes.

Next: archive the original JSON with exact software/hardware provenance, then
collect repeatable timings under controlled conditions or on another CUDA host.
