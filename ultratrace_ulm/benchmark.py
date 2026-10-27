"""Reproducible CUDA beamforming measurements in isolated subprocesses.

Run ``python -m ultratrace_ulm.benchmark --help``. Importing this module and
testing its protocol do not require CUDA. See BENCHMARKS.md for metric scopes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from unittest.mock import patch

import numpy as np

from . import beamform_core as core

RTOL = 2e-6
ATOL = 1e-6


@dataclass(frozen=True)
class Fixture:
    frames: int = 16
    angles: int = 4
    rows: int = 2
    cols: int = 4
    samples: int = 512
    elev_planes: int = 3
    seed: int = 3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be {'nonnegative' if name == 'seed' else 'positive'} integer")


def make_fixture(spec: Fixture):
    """Synthetic random IQ; no download and no assertion of physical realism."""
    rng = np.random.default_rng(spec.seed)
    shape = (spec.frames, spec.angles, spec.rows, spec.cols, spec.samples)
    iq = np.empty(shape, dtype=np.complex64)
    iq.real = rng.standard_normal(shape, dtype=np.float32)
    iq.imag = rng.standard_normal(shape, dtype=np.float32)
    config = core.NeutralConfig.from_h5_attrs(dict(
        sampling_rate_hz=20e6, tx_freq_hz=5e6, speed_of_sound_m_s=1540.,
        start_depth_m=.005, end_depth_m=.010, rx_offset_s=0.,
        element_pitch_x_m=.000208, element_pitch_y_m=.001664,
        ana_mux=False, num_elev_planes=spec.elev_planes, row_index=-1,
        mean_subtract_channels=False, large_fov=False,
        z_coarseness=1., x_coarseness=1., f_number=2.,
        tx_az_half_aperture=spec.cols,
    ))
    return iq, np.zeros((spec.angles, spec.cols)), np.zeros((spec.angles, spec.rows)), config


def fixture_metadata(spec, fixture):
    iq, _, _, config = fixture
    grid = core.build_grid(config)
    return {
        "parameters": asdict(spec), "iq_shape": list(iq.shape),
        "iq_dtype": str(iq.dtype), "iq_bytes": iq.nbytes,
        "iq_sha256": hashlib.sha256(memoryview(iq).cast("B")).hexdigest(),
        "grid_shape_z_elev_x": list(grid.x.shape), "config": asdict(config),
    }


def _cuda_module():
    try:
        import cupy as cp
        if not core.MACH_AVAILABLE:
            raise ImportError("mach.experimental.beamform is not importable")
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA devices available")
        return cp
    except Exception as exc:
        raise RuntimeError(
            "CUDA benchmark requires a supported CUDA host and the matching "
            "mach-cuda12 or mach-cuda13 extra. " + str(exc)
        ) from exc


class PoolMeter:
    """Observe the pool immediately after each allocation, including reuse.

    This measures rounded live/pool bytes, not requested bytes or whole-device
    memory. Non-CuPy allocators and CUDA context memory are outside its scope.
    """

    def __init__(self, pool):
        self.pool = pool
        self.live_peak = int(pool.used_bytes())
        self.reserved_peak = int(pool.total_bytes())

    def malloc(self, size):
        pointer = self.pool.malloc(size)
        self.live_peak = max(self.live_peak, int(self.pool.used_bytes()))
        self.reserved_peak = max(self.reserved_peak, int(self.pool.total_bytes()))
        return pointer

    def snapshot(self):
        return {
            "cupy_pool_live_peak_bytes": self.live_peak,
            "cupy_pool_reserved_peak_bytes": self.reserved_peak,
            "cupy_pool_live_after_bytes": int(self.pool.used_bytes()),
            "cupy_pool_reserved_after_bytes": int(self.pool.total_bytes()),
        }


def _peak_rss_bytes():
    # Linux reports KiB; macOS reports bytes. This is a process-lifetime high
    # water mark, not a delta or per-call allocation measurement.
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def measure_case(spec, *, n_chunks, warmups, repeats):
    core.validate_chunk_count(n_chunks)
    if warmups < 1 or repeats < 1:
        raise ValueError("warmups and repeats must be positive")
    cp = _cuda_module()
    fixture = make_fixture(spec)
    info = fixture_metadata(spec, fixture)
    sync = cp.cuda.runtime.deviceSynchronize
    warmup_times, repeat_times = [], []
    for timings, count in ((warmup_times, warmups), (repeat_times, repeats)):
        for _ in range(count):
            sync()
            start = time.perf_counter()
            output, grid = core.beamform_iq(*fixture, n_chunks=n_chunks)
            sync()
            timings.append(time.perf_counter() - start)
            del output, grid  # never hold a previous host image/grid during the next call

    # Keep instrumentation out of timed runs. Release idle pool blocks only
    # before this extra memory pass, after compilation/warm-up has completed.
    sync()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    meter = PoolMeter(pool)
    with cp.cuda.using_allocator(meter.malloc):
        output, _ = core.beamform_iq(*fixture, n_chunks=n_chunks)
        sync()
    memory = meter.snapshot()
    memory["process_peak_rss_bytes"] = _peak_rss_bytes()
    return {
        "n_chunks": n_chunks, "effective_chunks": min(n_chunks, spec.frames),
        "fixture": info, "output_shape_frames_elev_z_x": list(output.shape),
        "output_dtype": str(output.dtype), "warmup_seconds": warmup_times,
        "repeat_seconds": repeat_times, "median_seconds": statistics.median(repeat_times),
        "memory": memory,
    }, output


def host_reference(spec):
    """Host angle sums with one frame on device, outside measured workers.

    The reference must fit even when a measured single-batch run exhausts
    device memory. All settings use this same independent host composition.
    """
    cp = _cuda_module()
    kernel = core._mach_beamform
    accumulated = None
    calls = 0

    def record_angle(**kwargs):
        nonlocal accumulated, calls
        angle = kernel(**kwargs)
        host = cp.asnumpy(angle)
        if accumulated is None:
            accumulated = np.empty((host.shape[0], spec.frames), dtype=np.complex64)
        frame, angle_index = divmod(calls, spec.angles)
        if angle_index == 0:
            accumulated[:, frame] = host[:, 0]
        else:
            accumulated[:, frame] += host[:, 0]
        calls += 1
        return angle

    with patch.object(core, "_mach_beamform", side_effect=record_angle):
        output, grid = core.beamform_iq(*make_fixture(spec), n_chunks=spec.frames)
    if calls != spec.frames * spec.angles:
        raise RuntimeError("Reference did not receive every frame/angle")
    reference = accumulated.reshape(*grid.x.shape, spec.frames).transpose(3, 1, 0, 2)
    comparison = compare_outputs(output, reference)
    if not comparison["passed"]:
        raise RuntimeError(f"Single-frame batches failed host-reference validation: {comparison}")
    return np.ascontiguousarray(reference)


def compare_outputs(candidate, reference):
    result = {"passed": False, "rtol": RTOL, "atol": ATOL}
    if candidate.shape != reference.shape or candidate.ndim != 4:
        return {**result, "reason": "output shape mismatch"}
    if candidate.dtype != np.complex64 or reference.dtype != np.complex64:
        return {**result, "reason": "expected complex64 output"}
    passed, nonzero, max_error, error_energy, ref_energy = True, False, 0., 0., 0.
    # Frame-wise comparison also works with memory-mapped output files, without
    # allocating a second complete complex128 volume in the coordinator.
    for actual, expected in zip(candidate, reference):
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            return {**result, "reason": "nonfinite output"}
        nonzero = nonzero or bool(np.any(expected != 0))
        delta = actual.astype(np.complex128) - expected
        error = np.abs(delta)
        max_error = max(max_error, float(error.max(initial=0)))
        error_energy += float(np.sum(error ** 2))
        ref_energy += float(np.sum(np.abs(expected.astype(np.complex128)) ** 2))
        passed = passed and bool(np.all(error <= ATOL + RTOL * np.abs(expected)))
    result.update(passed=passed and nonzero, max_absolute_error=max_error,
                  relative_l2_error=float(np.sqrt(error_energy / ref_energy)) if ref_energy else None)
    if not nonzero:
        result["reason"] = "reference is all zero"
    return result


def _environment(cp):
    def version(package):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    device = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(device)
    name = props["name"]
    return {
        "platform": platform.platform(), "python": sys.version,
        "packages": {key: version(key) for key in (
            "ultratrace-ulm-pipeline", "numpy", "scipy", "mach-beamform", "torch")},
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "device": {"ordinal": int(device),
                   "name": name.decode() if isinstance(name, bytes) else name,
                   "total_memory_bytes": int(props["totalGlobalMem"]),
                   "compute_capability": [int(props["major"]), int(props["minor"])]},
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUPY_GPU_MEMORY_LIMIT", "CUPY_CACHE_DIR",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def _source_revision():
    root = Path(__file__).resolve().parent.parent

    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": status != "" if status is not None else None}


def _worker(request_path, response_path):
    request = json.loads(Path(request_path).read_text())
    if request.get("mode") not in ("reference", "measure"):
        raise ValueError(f"Unknown worker mode: {request.get('mode')}")
    spec = Fixture(**request["fixture"])
    environment = _environment(_cuda_module())
    if request["mode"] == "reference":
        output = host_reference(spec)
        result = {"method": "single-frame MACH batches with angles summed on host"}
    else:
        result, output = measure_case(spec, n_chunks=request["n_chunks"],
                                      warmups=request["warmups"], repeats=request["repeats"])
    output_path = Path(response_path).with_suffix(".npy")
    np.save(output_path, output, allow_pickle=False)
    result.update(status="ok", output_path=str(output_path), pid=os.getpid(), environment=environment)
    Path(response_path).write_text(json.dumps(result, allow_nan=False))


def _run_worker(request, directory):
    stem = Path(directory) / uuid.uuid4().hex
    request_path, response_path = stem.with_suffix(".request.json"), stem.with_suffix(".result.json")
    request_path.write_text(json.dumps(request))
    completed = subprocess.run(
        [sys.executable, "-m", "ultratrace_ulm.benchmark", "--_worker", str(request_path), str(response_path)],
        capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"Worker exited {completed.returncode}: {completed.stderr[-6000:]}")
    result = json.loads(response_path.read_text())
    # Preserve warnings without mixing them into the JSON transport.
    result["diagnostics"] = {"stdout": completed.stdout[-6000:], "stderr": completed.stderr[-6000:]}
    return result


def run_benchmark(spec, chunks, *, warmups, repeats):
    chunks = [core.validate_chunk_count(count) for count in chunks]
    if not chunks or warmups < 1 or repeats < 1:
        raise ValueError("chunks, warmups and repeats must be nonempty/positive")
    report = {
        "schema_version": 1, "status": "ok", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": _source_revision(), "fixture": asdict(spec),
        "settings": {"chunks": chunks, "warmups": warmups, "repeats": repeats}, "cases": [],
        "measurement_scope": {
            "timing": "synchronized wall time of beamform_iq; warm-up separate; memory instrumentation excluded",
            "device_memory": "CuPy pool only; extra warmed-kernel pass after freeing idle pool blocks; excludes other allocators and CUDA context",
            "host_memory": "worker process peak RSS through measurement, including imports, fixture and warm-up; before output serialization",
            "validation": "memory-pass output versus single-frame MACH batches with angles summed on host; synthetic IQ; no physical or downstream validation",
        },
    }
    request = {"fixture": asdict(spec), "warmups": warmups, "repeats": repeats}
    with tempfile.TemporaryDirectory(prefix="ulm-benchmark-") as directory:
        try:
            reference_info = _run_worker({**request, "mode": "reference"}, directory)
            reference = np.load(reference_info.pop("output_path"), mmap_mode="r", allow_pickle=False)
            report["reference"] = reference_info
        except (RuntimeError, OSError, ValueError) as exc:
            report.update(status="failed", reference={"status": "failed", "error": str(exc)})
            return report
        for count in chunks:
            try:
                case = _run_worker({**request, "mode": "measure", "n_chunks": count}, directory)
                candidate = np.load(case.pop("output_path"), mmap_mode="r", allow_pickle=False)
                case["comparison"] = compare_outputs(candidate, reference)
                del candidate
                if not case["comparison"]["passed"]:
                    case["status"] = "failed"
            except (RuntimeError, OSError, ValueError) as exc:
                case = {"status": "failed", "error": str(exc)}
            case["n_chunks"] = count
            report["cases"].append(case)
            if case["status"] != "ok":
                report["status"] = "failed"
        del reference
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_worker":
        _worker(*argv[1:])
        return 0

    def positive(value):
        number = int(value)
        if number < 1:
            raise argparse.ArgumentTypeError("must be positive")
        return number

    def nonnegative(value):
        number = int(value)
        if number < 0:
            raise argparse.ArgumentTypeError("must be nonnegative")
        return number

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination")
    parser.add_argument("--chunks", nargs="+", type=positive, default=[1, 2, 4, 8])
    parser.add_argument("--warmups", type=positive, default=1)
    parser.add_argument("--repeats", type=positive, default=5)
    for name, default in asdict(Fixture()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=nonnegative if name == "seed" else positive,
                            default=default)
    args = parser.parse_args(argv)
    spec = Fixture(**{key: getattr(args, key) for key in asdict(Fixture())})
    report = run_benchmark(spec, args.chunks, warmups=args.warmups, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Benchmark {report['status']}: {args.output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
