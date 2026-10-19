"""Run one legacy PMDFNet model for competition reproduction.

This wrapper deliberately calls the existing train_pipeline once instead of
executing train.py's historical multi-model loop.  It configures the data
root, seed, device, output directory, and core training hyperparameters via
environment variables consumed by train.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "dual_segformer_convnexttiny_chv1_add"
DEFAULT_WINDOWS_DATA_ROOT = r"Z:\huilin\bdd\cp_data\mmlsv2_mapped_mars_ls"
DEFAULT_LINUX_DATA_ROOT = "/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls"


def default_data_root() -> str:
    configured = os.environ.get("MARS_DATA_ROOT") or os.environ.get("DATA_ROOT")
    if configured:
        return configured
    return DEFAULT_WINDOWS_DATA_ROOT if os.name == "nt" else DEFAULT_LINUX_DATA_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one legacy PMDFNet model with a reproducible configuration."
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", DEFAULT_MODEL),
        help="Model name accepted by networks.get_model.",
    )
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", "42")))
    parser.add_argument(
        "--device",
        default=os.environ.get("TRAIN_DEVICE", "cuda:0"),
        help="Requested torch device; falls back to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("TRAIN_OUTPUT_ROOT", "outputs/reproduction"),
        help="Project-side staging directory used while training.",
    )
    parser.add_argument(
        "--final-output-root",
        default=os.environ.get("TRAIN_FINAL_OUTPUT_ROOT", ""),
        help="Dataset-side destination for the completed run; defaults to <data-root>/outputs.",
    )
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("TRAIN_EPOCHS", "100")))
    parser.add_argument("--batch-size", type=int, default=0, help="0 keeps the model-specific legacy default.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("TRAIN_NUM_WORKERS", "4")),
    )
    parser.add_argument(
        "--strict-determinism",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("TRAIN_STRICT_DETERMINISM", "1").lower()
        not in {"0", "false", "no", "off"},
        help="Raise on unsupported nondeterministic operators when enabled.",
    )
    parser.add_argument(
        "--conduct-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation and select checkpoints from validation metrics.",
    )
    return parser


def configure_environment(args: argparse.Namespace) -> None:
    data_root = str(Path(args.data_root).expanduser())
    required_dirs = [
        Path(data_root) / "train" / "images",
        Path(data_root) / "train" / "masks",
        Path(data_root) / "val" / "images",
        Path(data_root) / "val" / "masks",
    ]
    missing = [str(path) for path in required_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing required dataset directories:\n" + "\n".join(missing))

    # Set all supported names so a local .env file cannot silently redirect a
    # reproduction run to a different dataset.
    os.environ["MARS_DATA_ROOT"] = data_root
    os.environ["DATA_ROOT"] = data_root
    os.environ["MMLSV2_DATA_ROOT"] = data_root
    os.environ["TRAIN_SEED"] = str(args.seed)
    os.environ["TRAIN_DEVICE"] = args.device
    os.environ["TRAIN_OUTPUT_ROOT"] = str(Path(args.output_root).expanduser())
    final_output_root = args.final_output_root or str(Path(data_root) / "outputs")
    os.environ["TRAIN_FINAL_OUTPUT_ROOT"] = str(Path(final_output_root).expanduser())
    os.environ["TRAIN_FINALIZE_RESULTS"] = "1"
    os.environ["TRAIN_EPOCHS"] = str(args.epochs)
    os.environ["TRAIN_LR"] = str(args.lr)
    os.environ["TRAIN_WEIGHT_DECAY"] = str(args.weight_decay)
    os.environ["TRAIN_VAL_INTERVAL"] = str(args.val_interval)
    os.environ["TRAIN_NUM_WORKERS"] = str(args.num_workers)
    os.environ["TRAIN_STRICT_DETERMINISM"] = "1" if args.strict_determinism else "0"
    if args.batch_size > 0:
        os.environ["TRAIN_BATCH_SIZE"] = str(args.batch_size)
    else:
        os.environ.pop("TRAIN_BATCH_SIZE", None)


def main() -> None:
    args = build_parser().parse_args()
    configure_environment(args)
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    # Import only after environment setup: train.py also reads .env at import
    # time through env_utils.py.
    import train as legacy_train

    legacy_train.RANDOM_SEED = args.seed
    print("=== Competition reproduction run ===")
    print(f"model: {args.model_name}")
    print(f"data_root: {os.environ['MARS_DATA_ROOT']}")
    print(f"seed: {args.seed}")
    print(f"device: {args.device}")
    print(f"epochs: {args.epochs}")
    print(f"num_workers: {args.num_workers}")
    print(f"strict_determinism: {args.strict_determinism}")
    print(f"validation: {args.conduct_val}")
    legacy_train.train_pipeline(args.model_name, conduct_val=args.conduct_val)


if __name__ == "__main__":
    main()
