"""Run one seed of the frozen paper experiment protocol.

The shell entry point deliberately starts one Python process per seed so that
PYTHONHASHSEED and CUDA determinism are fixed before Python/PyTorch initialize.
"""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_ablation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a selected group from the MMLSv2 paper experiment protocol."
    )
    parser.add_argument("--config", required=True, help="Protocol JSON file.")
    parser.add_argument(
        "--group",
        default="core",
        choices=["core", "ablations", "extended", "single_channel", "all"],
        help=(
            "Experiment group to run; all includes every group in the JSON. "
            "The groups core, ablations, extended and single_channel are disjoint."
        ),
    )
    parser.add_argument("--seed", required=True, type=int, help="One reproducibility seed.")
    parser.add_argument("--data-root", default="", help="Override the protocol data root.")
    parser.add_argument("--device", default="", help="Override the protocol device, e.g. cuda:0.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for smoke/debug runs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch-size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument(
        "--experiments",
        default="",
        help="Optional comma-separated experiment IDs to run, e.g. M3,M4,L1.",
    )
    parser.add_argument(
        "--finalize-results",
        type=train_ablation.str2bool,
        default=None,
        help="Explicitly override whether to copy each completed run to the dataset outputs directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build/check data and exit before training.")
    parser.add_argument(
        "--skip-model-init",
        action="store_true",
        help="Skip model construction; useful for data-only smoke checks.",
    )
    return parser.parse_args(argv)


def load_protocol(path):
    with open(path, "r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not protocol.get("experiments"):
        raise ValueError("Protocol must contain a non-empty experiments list")
    if not isinstance(protocol.get("common", {}), dict):
        raise ValueError("Protocol common section must be a JSON object")
    return protocol


def build_config(protocol, experiment, args):
    merged = dict(protocol.get("common", {}))
    merged.update({key: value for key, value in experiment.items() if key in train_ablation.DEFAULTS})

    base_name = merged.get("experiment_name") or experiment.get("id") or "paper_experiment"
    merged["experiment_name"] = f"{base_name}_s{args.seed}"
    merged["seed"] = args.seed
    if args.data_root:
        merged["data_root"] = args.data_root
    if args.device:
        merged["device"] = args.device
    if args.epochs is not None:
        merged["epochs"] = args.epochs
    if args.batch_size is not None:
        merged["batch_size"] = args.batch_size
    if args.num_workers is not None:
        merged["num_workers"] = args.num_workers
    if args.finalize_results is not None:
        merged["finalize_results"] = args.finalize_results
    if args.dry_run:
        merged["dry_run"] = True
    if args.skip_model_init:
        merged["skip_model_init"] = True

    for key in train_ablation.DEFAULTS:
        merged.setdefault(key, train_ablation.DEFAULTS[key])
    if not merged.get("data_root"):
        raise ValueError(
            "No data root supplied. Set --data-root or MMLSV2_DATA_ROOT/MARS_DATA_ROOT/DATA_ROOT."
        )
    return train_ablation.RunConfig(
        **{key: merged[key] for key in train_ablation.DEFAULTS}
    )


def main(argv=None):
    args = parse_args(argv)
    protocol_path = Path(args.config).resolve()
    protocol = load_protocol(protocol_path)

    group_selected = [
        experiment
        for experiment in protocol["experiments"]
        if train_ablation.experiment_in_group(experiment, args.group)
    ]
    if not group_selected:
        raise ValueError(f"No experiments found for group: {args.group}")

    requested_ids = {
        item.strip()
        for item in args.experiments.split(",")
        if item.strip()
    }
    if requested_ids:
        selected = [
            experiment
            for experiment in group_selected
            if experiment.get("id", "") in requested_ids
        ]
        selected_ids = {experiment.get("id", "") for experiment in selected}
        missing_ids = sorted(requested_ids - selected_ids)
        if missing_ids:
            raise ValueError(
                "Requested experiment IDs are unavailable in the selected group: "
                + ", ".join(missing_ids)
            )
    else:
        selected = group_selected

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    print(
        f"Paper protocol={protocol_path} group={args.group} seed={args.seed} "
        f"experiments={len(selected)} "
        f"ids={','.join(experiment.get('id', '?') for experiment in selected)}"
    )

    run_dirs = []
    for experiment in selected:
        run_id = experiment.get("id", experiment.get("experiment_name", "unknown"))
        cfg = build_config(protocol, experiment, args)
        print(f"=== {run_id}: {cfg.experiment_name} ===")
        run_dir = train_ablation.run_experiment(cfg)
        run_dirs.append(str(run_dir))

    print("Completed runs:")
    for run_dir in run_dirs:
        print(run_dir)


if __name__ == "__main__":
    main()
