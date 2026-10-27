# CUDA beamforming benchmarks

The harness is implemented; hardware results are still pending. CPU tests check
the measurement protocol, not CUDA correctness, throughput, or total GPU memory.

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
for every child process. The command generates deterministic complex64 random
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
cold compilation measurement.

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
   values, a nonzero reference, and elementwise `rtol=2e-6, atol=1e-6`. Record
   maximum absolute and relative L2 error. The tolerance matches the existing
   opt-in CUDA composition test and still needs confirmation on real hardware.

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

Until actual CUDA reports have been collected and reviewed, retain the current
batch defaults and keep hardware validation marked pending in `ROADMAP.md`.
