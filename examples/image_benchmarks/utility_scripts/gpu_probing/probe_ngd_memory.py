"""Single- or multi-GPU memory/util probe for the image benchmark runner.

Steps the REAL training path (same model, FlowMatching wrapper, ImageTrainer,
NGD with CG linear_solver_maxiter=50, N GPUs with a DataParallelContext of N
devices, exactly as the production runner does) and measures - all in-process,
no nvidia-smi:

  - device_memory_peak_seen_max: max pynvml memory.used over the measurement
    window on the first GPU of the range. With PREALLOCATE=false the BFC arena
    grows on demand to cover the largest per-step transient, so this high-water
    is a truthful upper bound of the per-step peak (a hard OOM during the
    window is definitive "does not fit").
  - device_memory_used_steady_growth_in_window: whether memory.used was still
    climbing in the last third of the window (true peak may exceed the seen
    high-water).
  - pprof_live_bytes: retained JAX-array bytes from
    jax.profiler.device_memory_profile() (pprof proto), taken once after the
    window; ~parameter state only, a leak detector, not the transient peak.
  - pynvml SM utilization % (measurement window only, first GPU of the range).
  - mean per-step wall time (trainer.step blocks until ready) and
    time_per_sample_ms (= mean_step_seconds / batch_size * 1000).

One candidate runs per process (see --sweep), so XLA arena state cannot leak
between candidates and a hard OOM during the run is definitive for that batch
size.

Environment is derived from the CLI before any JAX import, so it is identical
across the sweep parent and its freshly spawned candidate processes:
  CUDA_VISIBLE_DEVICES = str(j) for j in [--gpu-index, --gpu-index+--gpu-count)
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  (MEM_FRACTION intentionally left at the default 0.75: a 1.0 fraction lets the
   arena grow to the full device during cuDNN/runtime autotune and produces
   spurious OOMs).

Config: use --config with any image-benchmark preset (controls the dataset and
the architecture/rhs). The method is always the worst-case NGD memory path
   (step_size 1e-3, linear regularization 1e-3, CG maxiter 50). The global batch
   size must divide the training split and must be divisible by --gpu-count,
   unless --allow-drop-last is given (then any batch <= the split size is
   measured; the training loop drops the remainder each epoch, matching the
   production loader's drop_last behavior).

Usage:
  # single candidate
  python probe_ngd_memory.py --config <preset>.json --batch 2000 \
      --warmup 8 --measure 20 --output r2000.json
  # full sweep, one process per candidate (production workflow)
  python probe_ngd_memory.py --config <preset>.json --sweep \
      --candidates 600,1000,1200,1500,2000,3000,4000 \
      --warmup 8 --measure 20 --output-dir probe_results --gpu-count 1
  # multi-GPU (global batch sharded over N GPUs)
  python probe_ngd_memory.py --config <preset>.json --sweep \
      --candidates 2000 --gpu-count 2 --gpu-index 2
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

# --- device selection BEFORE any JAX import ---------------------------------
# Memory measurement strategy (JAX 0.11):
#  * The pprof device_memory_profile only tracks persistent JAX arrays (~MB),
#    not executable-internal step temporaries.
#  * A preallocated arena (fraction N) pins memory.used to a constant and hides
#    the per-step transient peak.
#  * PREALLOCATE=false grows the BFC arena on demand to cover the biggest
#    transient of every step, so the pynvml memory.used high-water over the
#    measurement window is a truthful upper bound of the per-step peak, and a
#    hard OOM during the window is definitive "does not fit".
# One candidate runs per process (sweep mode), so arena state never leaks
# between candidates.
def _cli_int(flag: str, default: int) -> int:
    for index, token in enumerate(sys.argv):
        if token == flag:
            try:
                return int(sys.argv[index + 1])
            except (IndexError, ValueError):
                return default
    return default


_gpu_index = _cli_int("--gpu-index", 0)
_gpu_count = _cli_int("--gpu-count", 1)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(_gpu_index + offset) for offset in range(_gpu_count))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
os.environ.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
os.environ.pop("TF_GPU_ALLOCATOR", None)

_REPO_ROOT = Path(os.environ.get(
    "NEURIPP_REPO_ROOT", Path(__file__).resolve().parents[4])).resolve()
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from flow_matching_image_benchmark_runner import (  # noqa: E402
    _build_encoder,
    _make_stream,
    prepare_dataset,
)
from image_benchmarks.config import load_config, plan_runs  # noqa: E402
from image_benchmarks.distributed import DataParallelContext  # noqa: E402
from image_benchmarks.rhs.registry import build_rhs  # noqa: E402
from image_benchmarks.training.trainer import ImageTrainer  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from neuripp.parametric_pushforward.flow_matching import (  # noqa: E402
    FlowMatching,
    flow_matching_loss,
)

_SAMPLE_TYPE_KEY = "inuse_space"
_SPACE_TYPE_KEYS = ("inuse_space", "space", "bytes")


# ---------- pprof wire-format parsing (JAX 0.11 device_memory_profile) ------
def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes):
    """Yield (field_number, wire_type, value, next_pos) for a message."""
    pos = 0
    length = len(buf)
    while pos < length:
        key, pos = _read_varint(buf, pos)
        field = key >> 3
        wire = key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
            yield field, wire, value, pos
        elif wire == 1:
            yield field, wire, buf[pos:pos + 8], pos + 8
            pos += 8
        elif wire == 2:
            size, pos = _read_varint(buf, pos)
            payload = buf[pos:pos + size]
            yield field, wire, payload, pos + size
            pos += size
        elif wire == 5:
            yield field, wire, buf[pos:pos + 4], pos + 4
            pos += 4
        else:
            raise ValueError(f"unexpected protobuf wire type {wire}")


def _packed_varints(payload: bytes) -> list[int]:
    values = []
    pos = 0
    while pos < len(payload):
        value, pos = _read_varint(payload, pos)
        values.append(value)
    return values


def _profile_live_bytes(pprof_bytes: bytes) -> dict:
    """Sum the byte-valued sample slot from a (gzip'd) pprof memory profile."""
    try:
        if pprof_bytes[:2] == b"\x1f\x8b":
            import gzip
            pprof_bytes = gzip.decompress(pprof_bytes)
    except Exception:
        return {"sample_types": [], "live_bytes": None}
    string_table: list[str] = []
    sample_types: list[tuple[int | None, int | None]] = []
    sample_payloads: list[bytes] = []
    for field, wire, value, _ in _iter_fields(pprof_bytes):
        if wire != 2:
            continue
        if field == 6:  # string_table
            string_table.append(value.decode("utf-8", "replace"))
        elif field == 1:  # ValueType {int64 type=1; int64 unit=2}
            type_idx = unit_idx = None
            for sub_field, sub_wire, sub_value, _ in _iter_fields(value):
                if sub_wire != 0:
                    continue
                if sub_field == 1:
                    type_idx = sub_value
                elif sub_field == 2:
                    unit_idx = sub_value
            sample_types.append((type_idx, unit_idx))
        elif field == 2:  # Sample
            sample_payloads.append(value)
    value_index = None
    for index, (type_idx, _unit_idx) in enumerate(sample_types):
        if type_idx is not None and type_idx < len(string_table) \
                and string_table[type_idx] in _SPACE_TYPE_KEYS:
            value_index = index
            break
    if value_index is None:
        value_index = 0 if sample_types else None
    if value_index is None:
        return {"sample_types": string_table, "live_bytes": None}
    total = 0
    for payload in sample_payloads:
        values: list[int] = []
        for sub_field, sub_wire, sub_value, _ in _iter_fields(payload):
            if sub_field != 2:  # Sample.value
                continue
            if sub_wire == 2:
                values.extend(_packed_varints(sub_value))
            else:
                values.append(sub_value)
        if value_index < len(values):
            total += values[value_index]
    return {"sample_types": string_table, "live_bytes": total}


class UtilizationSampler:
    """Background pynvml sampler for SM utilization and memory.used.

    Samples the first GPU of the visible set (device_index is an NVML index
    into the physical devices; with CUDA_VISIBLE_DEVICES set to a contiguous
    range this equals the first assigned GPU). For data-parallel runs the
    per-GPU footprint is identical across the replicas, so sampling the first
    GPU is representative.
    """

    def __init__(self, device_index: int = 0, interval: float = 0.1):
        self.device_index = device_index
        self.interval = interval
        self.samples: list[tuple[int, int]] = []
        self.device_total_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            import pynvml
        except Exception:
            return
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.device_total_bytes = int(info.total)
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.samples.append((int(util.gpu), int(memory.used)))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {
                "pynvml_sm_percent_max": None,
                "pynvml_sm_percent_mean": None,
                "device_memory_peak_seen_max": None,
                "device_total_bytes": self.device_total_bytes,
            }
        sm = [sample[0] for sample in self.samples]
        memory = [sample[1] for sample in self.samples]
        third = max(1, len(memory) // 3)
        steady_growth = (
            float(np.mean(memory[-third:])) > 1.15 * float(np.mean(memory[:third]))
        )
        return {
            "pynvml_sm_percent_max": max(sm),
            "pynvml_sm_percent_mean": round(float(np.mean(sm)), 2),
            "device_memory_peak_seen_max": max(memory),
            "device_memory_used_first_third_mean": round(float(np.mean(memory[:third])), 1),
            "device_memory_used_last_third_mean": round(float(np.mean(memory[-third:])), 1),
            "device_memory_used_steady_growth_in_window": bool(steady_growth),
            "device_total_bytes": self.device_total_bytes,
        }


def _trace_rss(label: str):
    """Print current RSS when NEURIPP_PROBE_TRACE_RSS=1 (host-RAM debugging)."""
    if os.environ.get("NEURIPP_PROBE_TRACE_RSS") == "1":
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS"):
                    print(f"[rss] {label}: {line.strip().split()[1]} kB",
                          flush=True)
                    return


def run_candidate(config, run, manifest, batch_size: int,
                  warmup: int, measure: int, gpu_index: int,
                  gpu_count: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["training"]["batch_size"] = batch_size
    cfg["resources"]["gpus_per_run"] = gpu_count
    if batch_size % gpu_count:
        raise ValueError(
            f"batch size {batch_size} must be divisible by --gpu-count {gpu_count}"
        )
    _trace_rss("start run_candidate")

    encoder = _build_encoder(cfg, manifest, run["rng_seeds"]["encoder_sampling"])
    _trace_rss("encoder built")
    train_stream = _make_stream(
        cfg,
        manifest,
        encoder,
        "train",
        run["rng_seeds"]["dataset_shuffle"],
        train=True,
        sampling_seed=run["rng_seeds"]["encoder_sampling"],
        augmentation_seed=run["rng_seeds"]["augmentation"],
    )
    initial_batch = train_stream.next_batch()
    model_rngs = nnx.Rngs(
        default=run["seed"], params=run["seed"],
        dropout=run["rng_seeds"]["model_dropout"],
    )
    rhs = build_rhs(cfg["rhs"], tuple(cfg["resolved"]["state_shape"]), rngs=model_rngs)
    sampling = cfg["evaluation"]["sampling"]
    model = FlowMatching(
        rhs,
        model_rngs,
        cfg["training"]["batch_size"],
        ode_method=sampling["method"],
        ode_nstep_max=sampling["steps"],
        ode_kwargs=sampling.get("kwargs", {}),
    )
    training_rngs = nnx.Rngs(
        default=run["rng_seeds"]["fm_noise"],
        fm_noise=run["rng_seeds"]["fm_noise"],
        fm_time=run["rng_seeds"]["fm_time"],
        model_dropout=run["rng_seeds"]["model_dropout"],
    )
    context = DataParallelContext.create(expected_device_count=gpu_count)
    trainer = ImageTrainer(
        model,
        run["method"],
        flow_matching_loss,
        initial_batch,
        training_rngs,
        data_parallel=context,
        dataset_size=manifest.splits["train"].count,
    )
    _trace_rss("trainer built")
    if trainer.method.initialization_updates == 0:
        train_stream.load_state_dict({"epoch": 0, "batch_index": 0})

    # Warmup: compilation + cuDNN autotuning.
    for index in range(warmup):
        batch = train_stream.next_batch()
        trainer.step(batch)
        _trace_rss(f"warmup step {index} done")

    # Measurement window.
    sampler = UtilizationSampler(device_index=gpu_index)
    sampler.start()
    step_times = []
    for _ in range(measure):
        batch = train_stream.next_batch()
        start = time.perf_counter()
        trainer.step(batch)
        step_times.append(time.perf_counter() - start)
    sampler.stop()

    # Single pprof snapshot after the window (per-step device_memory_profile
    # calls disable XLA buffer reuse and distort the grow-on-demand arena).
    # Skipped when NEURIPP_PROBE_SKIP_PPROF=1: the serialized profile can be
    # multi-GB and this host has only 16 GB RAM.
    if os.environ.get("NEURIPP_PROBE_SKIP_PPROF") == "1":
        parsed = {"live_bytes": None, "sample_types": None}
    else:
        try:
            parsed = _profile_live_bytes(jax.profiler.device_memory_profile())
        except Exception:
            parsed = {"live_bytes": None, "sample_types": None}

    mean_step = float(np.mean(step_times))
    report = {
        "batch_size": batch_size,
        "gpu_count": gpu_count,
        "steps_per_epoch": manifest.splits["train"].count // batch_size,
        "mean_step_seconds": round(mean_step, 4),
        "median_step_seconds": round(float(np.median(step_times)), 4),
        "time_per_sample_ms": round(mean_step / batch_size * 1000.0, 4),
        "pprof_live_bytes": parsed.get("live_bytes"),
        "pprof_sample_types": parsed.get("sample_types"),
        "parameter_count": trainer.accounting()["parameter_count"],
        "optimizer_step": trainer.step_count,
        **sampler.summary(),
    }
    total = report.get("device_total_bytes")
    if total is not None:
        report["used_fraction_of_device_seen_max"] = round(
            (report["device_memory_peak_seen_max"] or 0) / int(total), 4
        )
    return report


def _ngd_method_config():
    return [{
        "name": "ngd",
        "n_restarts": 1,
        "kwargs": {
            "step_size": 0.001,
            "linear_solver_regularization": 0.001,
            "linear_solver_tolerance": 0.000001,
            "linear_solver_maxiter": 50,
        },
    }]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(_EXAMPLES_DIR / "image_benchmarks" / "configs"
                    / "fashion_mnist_unet_300epoch_adamw_ngd.json"),
    )
    parser.add_argument("--batch", type=int, default=None,
                        help="single candidate global batch size")
    parser.add_argument("--sweep", action="store_true",
                        help="sweep candidates, one process each")
    parser.add_argument("--candidates", default="600,1000,1200,1500,2000,3000,4000")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--measure", type=int, default=20)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="probe_results")
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="first GPU of the visible range (NVML/physical id)")
    parser.add_argument("--gpu-count", type=int, default=1,
                        help="GPUs per run; global batch is sharded over them")
    parser.add_argument("--allow-drop-last", action="store_true",
                        help="measure batches that do not divide the training "
                             "split (remainder is dropped per epoch, as in the "
                             "production drop_last loader)")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    config["methods"] = _ngd_method_config()
    runs = plan_runs(config)
    run = next(run for run in runs if run["method"]["name"] == "ngd")

    if args.sweep:
        candidates = [int(value) for value in args.candidates.split(",") if value]
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        for batch_size in candidates:
            name = f"batch_{batch_size:05d}.json"
            path = output_dir / name
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(args.gpu_index + offset) for offset in range(args.gpu_count))
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--config", str(config_path),
                 "--batch", str(batch_size), "--output", str(path),
                 "--warmup", str(args.warmup), "--measure", str(args.measure),
                 "--gpu-index", str(args.gpu_index),
                 "--gpu-count", str(args.gpu_count)]
                + (["--allow-drop-last"] if args.allow_drop_last else []),
                env=env,
                capture_output=True, text=True,
            )
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8"))
                reports.append(report)
                print(f"[sweep] {path.name}: rc={result.returncode} "
                      f"peak={report.get('device_memory_peak_seen_max')}", flush=True)
            else:
                reports.append({
                    "batch_size": batch_size, "failed": True,
                    "returncode": result.returncode,
                    "stderr_tail": (result.stderr or "").splitlines()[-5:],
                })
                print(f"[sweep] {name}: FAILED rc={result.returncode}", flush=True)
        summary = {
            "candidates": candidates,
            "warmup_steps": args.warmup,
            "measure_steps": args.measure,
            "gpu_index": args.gpu_index,
            "gpu_count": args.gpu_count,
            "allow_drop_last": args.allow_drop_last,
            "reports": reports,
        }
        output_path = output_dir / "summary.json"
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}", flush=True)
        return 0 if all(not report.get("failed") for report in reports) else 1

    if args.batch is None:
        parser.error("provide --batch or --sweep")
    batch_size = int(args.batch)
    manifest = prepare_dataset(config)
    steps_per_epoch = manifest.splits["train"].count // batch_size
    report = {"batch_size": batch_size}
    if not args.allow_drop_last and steps_per_epoch * batch_size != manifest.splits["train"].count:
        report["failed"] = True
        report["error"] = "batch size does not divide the training split"
    elif steps_per_epoch < 1:
        report["failed"] = True
        report["error"] = "batch size exceeds the training split size"
    else:
        report["steps_per_epoch"] = steps_per_epoch
        report["dropped_per_epoch"] = (
            manifest.splits["train"].count - steps_per_epoch * batch_size
        )
        try:
            report = run_candidate(
                config, run, manifest, batch_size,
                warmup=args.warmup, measure=args.measure,
                gpu_index=args.gpu_index, gpu_count=args.gpu_count,
            )
        except Exception as exc:
            report = {
                "batch_size": batch_size, "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if args.output:
        Path(args.output).expanduser().resolve().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 1 if report.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())