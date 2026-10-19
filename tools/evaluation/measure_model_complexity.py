"""Measure unified inference complexity for frozen paper-protocol runs.

This script evaluates the exact ``best.pth`` checkpoint of each selected run
and reports:

* trainable and total parameters;
* one-forward-pass GFLOPs (PyTorch profiler's supported-op estimate);
* synchronized latency and throughput/FPS;
* peak inference allocated/reserved CUDA memory.

The default set is the main comparison table: M1--M4, M6--M11 and L1 (the
stored run used for the paper-facing M5 main model).  The script intentionally
does not silently substitute a different checkpoint when a formal run is
missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.paper_run_utils import (  # noqa: E402
    DEFAULT_COMPLEXITY_IDS,
    load_protocol,
    parse_id_list,
    parse_seed_list,
    resolve_formal_run,
)


GB = 1024**3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure Params, GFLOPs, latency/FPS and peak inference VRAM."
    )
    parser.add_argument("--config", default="configs/mmlsv2_paper_protocol.json")
    parser.add_argument(
        "--run-root",
        default="outputs_experiments/pmdfnet_mmlsv2",
        help="Protocol output root containing summary.csv and timestamped run directories.",
    )
    parser.add_argument(
        "--ids",
        default=",".join(DEFAULT_COMPLEXITY_IDS),
        help="Comma-separated paper/protocol ids. Default: main baselines plus L1/M5.",
    )
    parser.add_argument("--seeds", default="41", help="Seed(s) to measure; default is seed 41.")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="outputs_experiments/complexity",
        help="Directory for complexity_results.json and complexity_results.csv.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record missing runs as skipped and continue; default is fail-fast.",
    )
    return parser.parse_args(argv)


def _as_first_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        for key in ("out", "logits", "main"):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    return output


def _synchronize(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_gb(value: int) -> float:
    return float(value) / GB


def _load_model(run: dict, device):
    import argparse as argparse_module
    import torch

    from tools.evaluation.evaluate_test_metrics import build_evaluation_model, load_checkpoint

    model_args = argparse_module.Namespace(
        run_config=run["config_path"],
        model_name="",
    )
    model, resolved_name, config_path, input_channels = build_evaluation_model(model_args)
    model = model.to(device)
    load_checkpoint(model, Path(run["checkpoint_path"]), device)
    model.eval()
    # Keep the checkpoint object and temporary tensors out of the timed region.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, resolved_name, config_path, input_channels


def _m3lsnet_backend(model_name: str) -> dict:
    if str(model_name).lower() != "m3lsnet":
        return {}
    try:
        import importlib

        module = importlib.import_module("networks.M3LSNet")

        available = module.selective_scan_fn is not None
        return {
            "m3lsnet_backend": (
                "mamba_ssm_selective_scan" if available else "gated_conv_simulator"
            ),
            "mamba_ssm_available": bool(available),
        }
    except Exception as exc:  # pragma: no cover - defensive metadata only
        return {"m3lsnet_backend": "backend_detection_failed", "backend_error": str(exc)}


def _profile_forward(model, data, device) -> tuple[float | None, str]:
    """Return total forward GFLOPs and the profiler status.

    ``torch.profiler`` only reports FLOPs for operators for which PyTorch has a
    FLOP formula.  Unsupported custom/selective-scan operators are therefore
    explicitly reported as undercounted instead of being presented as exact.
    """
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
        ) as profiler:
            with torch.inference_mode():
                _as_first_tensor(model(data))
        total_flops = 0
        for event in profiler.key_averages():
            total_flops += int(getattr(event, "flops", 0) or 0)
        if total_flops <= 0:
            return None, "no_supported_operator_flops"
        return float(total_flops), "torch_profiler_supported_ops"
    except Exception as exc:  # pragma: no cover - depends on torch build/model ops
        return None, f"profiler_failed:{type(exc).__name__}:{exc}"


def measure_run(run: dict, device, args) -> dict:
    import torch

    started = time.perf_counter()
    result = {
        "status": "failed",
        "paper_id": run["paper_id"],
        "stored_id": run["stored_id"],
        "seed": run["seed"],
        "experiment_name": run["experiment_name"],
        "run_dir": run["run_dir"],
        "checkpoint": run["checkpoint_path"],
        "batch_size": args.batch_size,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "device_requested": str(device),
    }
    model = None
    try:
        model, model_name, config_path, input_channels = _load_model(run, device)
        config_payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        input_size = int(config_payload.get("input_size", 128))
        input_shape = [args.batch_size, len(input_channels), input_size, input_size]
        data = torch.randn(*input_shape, device=device)
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

        model.eval()
        with torch.inference_mode():
            for _ in range(max(args.warmup, 0)):
                _as_first_tensor(model(data))
        _synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode():
                for _ in range(max(args.iterations, 1)):
                    _as_first_tensor(model(data))
            end.record()
            _synchronize(device)
            elapsed_ms_total = float(start.elapsed_time(end))
            peak_allocated_gb = _memory_gb(torch.cuda.max_memory_allocated(device))
            peak_reserved_gb = _memory_gb(torch.cuda.max_memory_reserved(device))
            device_name = torch.cuda.get_device_name(device)
        else:
            start_time = time.perf_counter()
            with torch.inference_mode():
                for _ in range(max(args.iterations, 1)):
                    _as_first_tensor(model(data))
            elapsed_ms_total = (time.perf_counter() - start_time) * 1000.0
            peak_allocated_gb = None
            peak_reserved_gb = None
            device_name = platform.processor() or "CPU"

        iterations = max(args.iterations, 1)
        latency_ms = elapsed_ms_total / iterations
        flops, flops_status = _profile_forward(model, data, device)
        gflops_per_image = None if flops is None else flops / args.batch_size / 1e9
        result.update(
            {
                "status": "completed",
                "model_name": model_name,
                "run_config": config_path,
                "input_channels": input_channels,
                "input_shape": input_shape,
                "device": str(device),
                "device_name": device_name,
                "parameters": int(total_params),
                "parameters_m": float(total_params / 1e6),
                "trainable_parameters": int(trainable_params),
                "trainable_parameters_m": float(trainable_params / 1e6),
                "gflops_per_image": gflops_per_image,
                "flops_total_for_batch": flops,
                "flops_backend": flops_status,
                "latency_ms_per_batch": float(latency_ms),
                "latency_ms_per_image": float(latency_ms / args.batch_size),
                "fps_images_per_second": float(args.batch_size * 1000.0 / latency_ms),
                "peak_inference_allocated_gb": peak_allocated_gb,
                "peak_inference_reserved_gb": peak_reserved_gb,
                "measurement_definition": {
                    "mode": "eval + inference_mode",
                    "latency": "synchronized forward-only wall time after warmup",
                    "fps": "batch_size / latency per batch",
                    "gflops": "torch.profiler supported forward operators; custom unsupported ops may be omitted",
                    "vram": "peak CUDA allocated/reserved memory during timed forward",
                },
                **_m3lsnet_backend(config_payload.get("model_name", model_name)),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def write_outputs(output_dir: Path, records: list[dict], args, protocol_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "completed" if records and all(item["status"] == "completed" for item in records) else "partial",
        "evaluation_type": "unified_model_complexity",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "run_root": str(Path(args.run_root).expanduser().resolve()),
        "ids": [item["paper_id"] for item in records],
        "seeds": sorted({item["seed"] for item in records}),
        "records": records,
        "material_passport": {
            "verification_status": "EXECUTED",
            "measurement_scope": "frozen best.pth checkpoints selected from formal summary.csv",
            "data_used": "none; synthetic tensor for inference measurement",
        },
    }
    (output_dir / "complexity_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for record in records for key in record})
    with (output_dir / "complexity_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError("batch-size and iterations must be positive; warmup cannot be negative")
    protocol_path = Path(args.config).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    ids = parse_id_list(args.ids, DEFAULT_COMPLEXITY_IDS)
    seeds = parse_seed_list(args.seeds, (41,))

    import torch

    requested = args.device
    device = torch.device(requested if requested else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device} but CUDA is unavailable")

    records = []
    for experiment_id in ids:
        for seed in seeds:
            try:
                run = resolve_formal_run(args.run_root, protocol, experiment_id, seed)
                print(f"[{experiment_id}/s{seed}] measuring {run['checkpoint_path']}", flush=True)
                record = measure_run(run, device, args)
            except Exception as exc:
                if not args.allow_missing:
                    raise
                record = {
                    "status": "skipped",
                    "paper_id": experiment_id,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    write_outputs(Path(args.output_dir).expanduser().resolve(), records, args, protocol_path)
    failed = [record for record in records if record["status"] != "completed"]
    print(f"Wrote {len(records)} complexity records; incomplete={len(failed)}")
    return 1 if failed and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

