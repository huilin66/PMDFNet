"""Shared reproducibility helpers for all experiment entry points.

Exact repeatability is guaranteed only when the same code commit, dataset
files, Python environment, GPU, and runtime configuration are used.  This
module makes the controllable sources of randomness explicit and records the
effective settings in each run.
"""

from __future__ import annotations

import os
import platform
import random
import sys
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def prepare_reproducibility_environment(seed: int) -> None:
    """Set process-level variables before CUDA work starts.

    ``PYTHONHASHSEED`` cannot retroactively change the current interpreter's
    hash randomization.  Launchers therefore set it before invoking Python;
    this function still records the intended value and protects direct
    programmatic entry points as far as Python permits.
    """

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_global_seed(
    seed: int,
    deterministic: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Seed Python/NumPy/PyTorch and configure deterministic execution."""

    prepare_reproducibility_environment(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    if deterministic:
        # Strict mode raises at the first unsupported nondeterministic op,
        # which is preferable for final paper runs.  Audit/smoke runs can use
        # warn_only while the offending operator is being identified.
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    else:
        torch.use_deterministic_algorithms(False)

    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "strict_determinism": bool(strict),
        "deterministic_algorithms_warn_only": bool(deterministic and not strict),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_tf32": bool(
            getattr(getattr(torch.backends, "cuda", None), "matmul", None)
            and torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }


def seed_worker(_: int) -> None:
    """Seed NumPy/Python RNGs inside a PyTorch DataLoader worker."""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def runtime_metadata() -> dict[str, Any]:
    """Return lightweight environment information for run manifests."""

    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        metadata["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return metadata


def file_sha256(path: str | os.PathLike[str]) -> str | None:
    """Hash a small manifest/config file when it exists."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_metadata(repo_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return the current Git commit and dirty state if available."""

    root = str(repo_root or Path(__file__).resolve().parent)

    def git_value(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "dirty": bool(git_value("status", "--short")),
    }
