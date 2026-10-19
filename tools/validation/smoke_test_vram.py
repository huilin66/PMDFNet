"""Measure one training step of every paper configuration under a VRAM cap.

The parent process does not initialize CUDA.  It starts one child process per
configuration so allocator state and peak-memory statistics are isolated.
Inputs are synthetic; no dataset files are read and no checkpoint is written.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_ablation


GB = 1024**3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure peak CUDA memory for one safe synthetic training step per protocol configuration."
    )
    parser.add_argument("--config", default="configs/mmlsv2_paper_protocol.json")
    parser.add_argument(
        "--group",
        choices=[
            "core",
            "ablations",
            "extended",
            "single_channel",
            "modality_grouping",
            "all",
        ],
        default="all",
        help=(
            "Protocol group to probe; all probes every listed experiment. "
            "The groups core, ablations, extended, single_channel and "
            "modality_grouping are disjoint when used with their corresponding "
            "protocol files."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Synthetic probe batch size. 8 is the conservative default; use 16 to probe the training batch.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-memory-gb", type=float, default=40.0)
    parser.add_argument(
        "--allocator-limit-gb",
        type=float,
        default=38.0,
        help="PyTorch per-process allocator limit. Keep below max-memory-gb for headroom.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output-dir", default="outputs_experiments/smoke/vram")
    parser.add_argument("--data-root", default="", help="Recorded only; synthetic probes do not read data.")
    parser.add_argument("--worker-config", default="", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def load_protocol(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not protocol.get("experiments"):
        raise ValueError("Protocol must contain a non-empty experiments list")
    return protocol


def in_group(experiment: dict, group: str) -> bool:
    # Single source of truth lives in train_ablation so the trainer, this probe
    # and the group launcher scripts always agree on what a group contains.
    return train_ablation.experiment_in_group(experiment, group)


def build_smoke_config(protocol: dict, experiment: dict, args) -> train_ablation.RunConfig:
    merged = dict(protocol.get("common", {}))
    merged.update(
        {
            key: value
            for key, value in experiment.items()
            if key in train_ablation.DEFAULTS
        }
    )
    base_name = merged.get("experiment_name") or experiment.get("id", "experiment")
    merged.update(
        {
            "data_root": args.data_root or merged.get("data_root") or "synthetic_smoke",
            "experiment_name": f"{base_name}_vram_smoke",
            "seed": args.seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "pretrain": False,
            "mixed_precision": False,
            "num_workers": 0,
            "mosaic_prob": 0.0,
            "summary_csv": "",
            "finalize_results": False,
            "dry_run": False,
            "skip_model_init": False,
        }
    )
    for key, value in train_ablation.DEFAULTS.items():
        merged.setdefault(key, value)
    return train_ablation.RunConfig(
        **{key: merged[key] for key in train_ablation.DEFAULTS}
    )


def memory_gb(value: int) -> float:
    return float(value) / GB


def current_memory(device_index: int) -> dict[str, float]:
    import torch

    return {
        "allocated_gb": memory_gb(torch.cuda.memory_allocated(device_index)),
        "reserved_gb": memory_gb(torch.cuda.memory_reserved(device_index)),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_worker(cfg: train_ablation.RunConfig, args) -> dict:
    import torch

    result = {
        "status": "failed",
        "experiment_name": cfg.experiment_name,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "max_memory_gb": args.max_memory_gb,
        "allocator_limit_gb": args.allocator_limit_gb,
        "config": asdict(cfg),
    }
    model = None
    optimizer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device(cfg.device)
        if device.type != "cuda":
            raise ValueError(f"VRAM smoke test requires a CUDA device, got {cfg.device}")
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(device_index)
        properties = torch.cuda.get_device_properties(device_index)
        total_memory_gb = memory_gb(properties.total_memory)
        allocator_limit_gb = min(args.allocator_limit_gb, args.max_memory_gb)
        if allocator_limit_gb <= 0:
            raise ValueError("allocator_limit_gb must be positive")
        fraction = min(allocator_limit_gb / total_memory_gb, 0.95)
        torch.cuda.set_per_process_memory_fraction(fraction, device_index)
        train_ablation.set_seed(cfg.seed, deterministic=True)
        torch.set_num_threads(1)

        free_before, _ = torch.cuda.mem_get_info(device_index)
        result.update(
            {
                "device": str(device),
                "device_index": device_index,
                "device_name": properties.name,
                "total_memory_gb": total_memory_gb,
                "free_memory_before_gb": memory_gb(free_before),
                "allocator_fraction": fraction,
            }
        )

        model = train_ablation.build_model(cfg).to(device)
        criterion = train_ablation.build_loss(cfg).to(device)
        optimizer = train_ablation.build_optimizer(cfg, model)
        model_parameters = sum(parameter.numel() for parameter in model.parameters())
        model_memory = current_memory(device_index)
        result.update(
            {
                "parameters_m": model_parameters / 1e6,
                "model_init_allocated_gb": model_memory["allocated_gb"],
                "model_init_reserved_gb": model_memory["reserved_gb"],
            }
        )

        if cfg.arch.lower() in {"dual_segformer", "pmdfnet"} or cfg.arch.lower().startswith(
            ("dual_", "pmdfnet_")
        ):
            input_channels = cfg.in_channels
        elif cfg.model_name and cfg.model_name.lower() not in {"auto", "custom"}:
            input_channels = cfg.in_channels
        else:
            input_channels = len(train_ablation.parse_channels(cfg.input_channels))
        input_shape = [cfg.batch_size, input_channels, cfg.input_size, cfg.input_size]
        data = torch.randn(*input_shape, device=device)
        target = torch.randint(
            low=0,
            high=cfg.num_classes,
            size=(cfg.batch_size, cfg.input_size, cfg.input_size),
            device=device,
        )
        result["input_shape"] = input_shape

        torch.cuda.reset_peak_memory_stats(device_index)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        output, loss = train_ablation.forward_loss(
            model, criterion, data, target, mixed_precision=False, device_type="cuda"
        )
        torch.cuda.synchronize(device_index)
        result["forward_peak_allocated_gb"] = memory_gb(
            torch.cuda.max_memory_allocated(device_index)
        )

        loss.backward()
        torch.cuda.synchronize(device_index)
        result["backward_peak_allocated_gb"] = memory_gb(
            torch.cuda.max_memory_allocated(device_index)
        )

        optimizer.step()
        torch.cuda.synchronize(device_index)
        result["step_peak_allocated_gb"] = memory_gb(
            torch.cuda.max_memory_allocated(device_index)
        )
        result["step_peak_reserved_gb"] = memory_gb(
            torch.cuda.max_memory_reserved(device_index)
        )
        result["peak_allocated_gb"] = result["step_peak_allocated_gb"]
        result["peak_reserved_gb"] = result["step_peak_reserved_gb"]
        result["cap_exceeded"] = result["peak_allocated_gb"] > args.max_memory_gb
        result["status"] = "cap_exceeded" if result["cap_exceeded"] else "completed"
    except RuntimeError as error:
        message = str(error)
        result["status"] = "memory_cap_hit" if "out of memory" in message.lower() else "failed"
        result["error"] = message[-4000:]
    except Exception as error:  # Keep one bad configuration from hiding the others.
        result["status"] = "failed"
        result["error"] = repr(error)[-4000:]
    finally:
        del optimizer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    return result


def worker_main(args) -> int:
    with Path(args.worker_config).open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    cfg = train_ablation.RunConfig(**config_data)
    result = run_worker(cfg, args)
    write_json(Path(args.result_json), result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


CSV_FIELDS = [
    "id",
    "group",
    "experiment_name",
    "status",
    "seed",
    "batch_size",
    "parameters_m",
    "input_shape",
    "device_name",
    "total_memory_gb",
    "free_memory_before_gb",
    "model_init_allocated_gb",
    "model_init_reserved_gb",
    "forward_peak_allocated_gb",
    "backward_peak_allocated_gb",
    "step_peak_allocated_gb",
    "peak_reserved_gb",
    "max_memory_gb",
    "allocator_limit_gb",
    "cap_exceeded",
    "error",
]


def parent_main(args) -> int:
    protocol_path = Path(args.config).resolve()
    protocol = load_protocol(protocol_path)
    selected = [experiment for experiment in protocol["experiments"] if in_group(experiment, args.group)]
    if not selected:
        raise ValueError(f"No experiments found for group: {args.group}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.allocator_limit_gb > args.max_memory_gb:
        raise ValueError("allocator-limit-gb must not exceed max-memory-gb")

    output_dir = Path(args.output_dir).resolve()
    workers_dir = output_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for experiment in selected:
        experiment_id = experiment.get("id", experiment.get("experiment_name", "experiment"))
        cfg = build_smoke_config(protocol, experiment, args)
        config_path = workers_dir / f"{experiment_id}.config.json"
        result_path = workers_dir / f"{experiment_id}.result.json"
        log_path = workers_dir / f"{experiment_id}.log"
        write_json(config_path, asdict(cfg))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-config",
            str(config_path),
            "--result-json",
            str(result_path),
            "--max-memory-gb",
            str(args.max_memory_gb),
            "--allocator-limit-gb",
            str(args.allocator_limit_gb),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONHASHSEED": str(args.seed),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TRAIN_STRICT_DETERMINISM": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        print(f"=== VRAM smoke: {experiment_id}, batch={args.batch_size} ===")
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            log_path.write_text(
                f"returncode={completed.returncode}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}",
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired as error:
            result = {
                "id": experiment_id,
                "group": experiment.get("group", "core"),
                "experiment_name": cfg.experiment_name,
                "status": "timeout",
                "seed": args.seed,
                "batch_size": args.batch_size,
                "max_memory_gb": args.max_memory_gb,
                "allocator_limit_gb": args.allocator_limit_gb,
                "error": f"timeout after {args.timeout_seconds}s: {error}",
            }
        else:
            if result_path.exists():
                with result_path.open("r", encoding="utf-8") as handle:
                    result = json.load(handle)
            else:
                result = {
                    "status": "worker_failed",
                    "error": f"worker returned {completed.returncode} without a result file",
                }
            result["id"] = experiment_id
            result["group"] = experiment.get("group", "core")
        results.append(result)
        peak = result.get("peak_allocated_gb", "n/a")
        print(f"{experiment_id}: {result.get('status')} | peak_allocated_gb={peak}")

    completed_at = datetime.now(timezone.utc).isoformat()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        commit = ""
    payload = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "validate",
            "verification_status": "CODE_ONLY_UNTIL_EXECUTED",
            "created_at": completed_at,
        },
        "protocol_file": str(protocol_path),
        "protocol_version": protocol.get("protocol_version", ""),
        "git_commit": commit,
        "device": args.device,
        "group": args.group,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_memory_gb": args.max_memory_gb,
        "allocator_limit_gb": args.allocator_limit_gb,
        "results": results,
    }
    write_json(output_dir / "vram_smoke_results.json", payload)
    with (output_dir / "vram_smoke_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = dict(result)
            if isinstance(row.get("input_shape"), list):
                row["input_shape"] = "x".join(str(value) for value in row["input_shape"])
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    print(f"Results: {output_dir / 'vram_smoke_results.csv'}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker_config:
        return worker_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
