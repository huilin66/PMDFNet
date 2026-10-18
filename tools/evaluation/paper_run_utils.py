"""Helpers for locating and describing frozen paper-protocol runs.

The training launcher writes one directory per ``experiment_id``/seed and a
row to ``summary.csv``.  Evaluation utilities should resolve those records in
exactly the same way instead of guessing a checkpoint from a glob.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


DEFAULT_COMPLEXITY_IDS = ("M1", "M2", "M3", "M4", "M6", "M7", "M8", "M9", "M10", "M11", "L1")


def parse_id_list(value: str | None, default: tuple[str, ...]) -> list[str]:
    """Parse a comma-separated experiment list while preserving user order."""
    if value is None or not value.strip():
        return list(default)
    result = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in result:
            result.append(item)
    if not result:
        raise ValueError("The experiment list must contain at least one id")
    return result


def parse_seed_list(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None or not value.strip():
        return list(default)
    result = []
    for item in value.replace(",", " ").split():
        seed = int(item)
        if seed not in result:
            result.append(seed)
    if not result:
        raise ValueError("The seed list must contain at least one seed")
    return result


def load_protocol(path: str | Path) -> dict:
    protocol_path = Path(path).expanduser().resolve()
    with protocol_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload.get("experiments"):
        raise ValueError(f"Protocol has no experiments: {protocol_path}")
    payload["_path"] = str(protocol_path)
    return payload


def protocol_experiment(protocol: dict, experiment_id: str) -> tuple[str, dict]:
    """Return the stored run id and protocol entry for a paper-facing id."""
    stored_id = protocol.get("paper_id_map", {}).get(experiment_id, experiment_id)
    for experiment in protocol.get("experiments", []):
        if experiment.get("id") == stored_id:
            return stored_id, experiment
    known = ", ".join(str(item.get("id")) for item in protocol.get("experiments", []))
    raise KeyError(f"Unknown experiment id {experiment_id!r}; stored ids: {known}")


def _read_summary(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.is_file():
        return []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_formal_run(
    run_root: str | Path,
    protocol: dict,
    experiment_id: str,
    seed: int,
) -> dict:
    """Resolve one completed formal run and its best checkpoint.

    The summary row is preferred because it records the exact timestamped run
    directory.  A narrow directory fallback is retained for older output
    trees that predate summary.csv.
    """
    root = Path(run_root).expanduser().resolve()
    stored_id, experiment = protocol_experiment(protocol, experiment_id)
    experiment_name = experiment.get("experiment_name")
    if not experiment_name:
        raise ValueError(f"Protocol entry {stored_id} has no experiment_name")
    expected_name = f"{experiment_name}_s{seed}"

    summary_path = root / "summary.csv"
    rows = _read_summary(summary_path)
    candidates: list[Path] = []
    for row in rows:
        if row.get("experiment_name") != expected_name:
            continue
        if row.get("status") not in {"", "completed", None}:
            continue
        raw_run_dir = row.get("run_dir", "")
        if not raw_run_dir:
            continue
        candidate = Path(raw_run_dir).expanduser()
        if not candidate.is_absolute():
            # summary.csv stores paths relative to the repository root.
            repo_root = root.parent.parent
            candidate = repo_root / candidate
        candidates.append(candidate.resolve())

    if not candidates:
        candidates = [path.resolve() for path in root.glob(f"{expected_name}_*") if path.is_dir()]

    valid = []
    for candidate in candidates:
        config_path = candidate / "config.json"
        checkpoint_candidates = (
            candidate / "checkpoints" / "best.pth",
            candidate / "best.pth",
        )
        checkpoint_path = next(
            (path for path in checkpoint_candidates if path.is_file()),
            None,
        )
        if config_path.is_file() and checkpoint_path is not None:
            valid.append((candidate, checkpoint_path))
    if not valid:
        raise FileNotFoundError(
            f"No completed checkpoint found for {experiment_id} (stored {stored_id}), seed {seed} "
            f"under {root}"
        )

    # If a historical summary contains duplicate rows, use the newest run
    # directory and expose the ambiguity to the caller.
    valid.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    run_dir, checkpoint_path = valid[0]
    row = next(
        (
            item
            for item in rows
            if item.get("experiment_name") == expected_name
            and Path(item.get("run_dir", "")).name == run_dir.name
        ),
        {},
    )
    return {
        "paper_id": experiment_id,
        "stored_id": stored_id,
        "seed": int(seed),
        "experiment_name": experiment_name,
        "run_dir": str(run_dir),
        "config_path": str(run_dir / "config.json"),
        "checkpoint_path": str(checkpoint_path),
        "summary_path": str(summary_path),
        "summary_row": row,
        "duplicate_candidates": [str(item[0]) for item in valid],
    }

