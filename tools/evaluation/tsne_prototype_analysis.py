"""Prototype-level t-SNE analysis for frozen PMDFNet checkpoints.

For each selected image and semantic class, this script computes one mean
feature vector over the GT region and one over the predicted region.  It then
embeds the resulting prototypes jointly for the requested models.  The output
keeps GT and Pred sources separate, matching the analysis intent of the MRSN
prototype script while using PMDFNet's PyTorch model and mapped MMLSv2 data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.evaluate_test_metrics import (  # noqa: E402
    build_evaluation_model,
    load_checkpoint,
)
from tools.evaluation.interpretability_common import (  # noqa: E402
    MappedLabeledDataset,
    SpatialHook,
    as_first_tensor,
    as_feature_tensor,
    find_spatial_layer,
    resize_mask,
    safe_name,
    save_json,
)
from tools.evaluation.paper_run_utils import (  # noqa: E402
    load_protocol,
    parse_id_list,
    resolve_formal_run,
)


DEFAULT_MODEL_IDS = ("M4", "L1")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract GT/Pred feature prototypes and create comparable t-SNE plots."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-ids", default="M4,L1")
    parser.add_argument("--run-root", default="outputs_experiments/pmdfnet_mmlsv2")
    parser.add_argument("--config", default="configs/mmlsv2_paper_protocol.json")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--feature-layer", default="auto", help="Named module or auto decoder-feature selection.")
    parser.add_argument("--class-ids", default="0,1")
    parser.add_argument("--selection-seed", type=int, default=20261018)
    parser.add_argument("--max-images-per-class", type=int, default=80)
    parser.add_argument("--max-prototypes", type=int, default=0, help="Optional total cap after extraction; 0 means all.")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs_experiments/interpretability/tsne")
    parser.add_argument("--resize", type=int, default=0)
    return parser.parse_args(argv)


def resolve_model(model_id: str, args: argparse.Namespace, device: torch.device):
    protocol = load_protocol(Path(args.config).expanduser().resolve())
    run = resolve_formal_run(args.run_root, protocol, model_id, args.seed)
    model_args = argparse.Namespace(run_config=run["config_path"], model_name="")
    model, resolved_name, config_path, input_channels = build_evaluation_model(model_args)
    model = model.to(device)
    load_checkpoint(model, Path(run["checkpoint_path"]), device)
    model.eval()
    config_payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return run, model, resolved_name, config_payload, input_channels


def normalization_for(config_payload: dict, split: str) -> str:
    if str(config_payload.get("dataset_type", "mars_ls")).lower() == "mmlsv2":
        return "none"
    return "test" if split == "test" else str(config_payload.get("normalization", "train"))


def resize_for(config_payload: dict, split: str, requested: int) -> int | None:
    if requested > 0:
        return requested
    return int(config_payload.get("input_size", 128)) if split == "val" else None


def selected_indices(dataset: MappedLabeledDataset, args: argparse.Namespace) -> list[int]:
    """Select the same image IDs for every model without using predictions."""

    if args.max_images_per_class <= 0:
        return list(range(len(dataset)))
    class_candidates: dict[int, list[int]] = {class_id: [] for class_id in args.class_ids}
    for index in range(len(dataset)):
        _, target, _ = dataset[index]
        target_np = target.numpy()
        for class_id in args.class_ids:
            if bool((target_np == class_id).any()):
                class_candidates[class_id].append(index)
    rng = np.random.default_rng(args.selection_seed)
    selected: set[int] = set()
    for class_id, candidates in class_candidates.items():
        candidates = list(candidates)
        rng.shuffle(candidates)
        selected.update(candidates[: args.max_images_per_class])
    return sorted(selected)


def model_prototypes(
    model_id: str,
    model: torch.nn.Module,
    dataset: MappedLabeledDataset,
    layer_name: str,
    layer: torch.nn.Module,
    indices: list[int],
    class_ids: list[int],
    device: torch.device,
) -> tuple[list[np.ndarray], list[dict]]:
    features: list[np.ndarray] = []
    records: list[dict] = []
    hook = SpatialHook(layer, with_gradient=False)
    try:
        with torch.inference_mode():
            for index in indices:
                image, target, image_name = dataset[index]
                hook.clear()
                logits = as_first_tensor(model(image.unsqueeze(0).to(device)))
                prediction = torch.argmax(logits, dim=1)[0]
                if hook.activation is None:
                    raise RuntimeError(f"Feature hook did not capture activation at {layer_name}")
                feature = as_feature_tensor(hook.activation)[0].float()
                for source_name, mask in (("gt", target.to(device)), ("pred", prediction)):
                    feature_mask = resize_mask(mask, tuple(feature.shape[-2:]))
                    for class_id in class_ids:
                        class_mask = feature_mask & (resize_mask(mask == class_id, tuple(feature.shape[-2:])))
                        pixel_count = int(class_mask.sum().item())
                        if pixel_count == 0:
                            continue
                        prototype = feature[:, class_mask].mean(dim=1).detach().cpu().numpy().astype(np.float32)
                        features.append(prototype)
                        records.append(
                            {
                                "model_id": model_id,
                                "image": image_name,
                                "index": index,
                                "source": source_name,
                                "class_id": class_id,
                                "feature_dim": int(prototype.shape[0]),
                                "feature_pixels": pixel_count,
                            }
                        )
    finally:
        hook.close()
    return features, records


def align_features(feature_lists: dict[str, list[np.ndarray]], pca_dim: int, seed: int):
    dimensions = {array.shape[0] for values in feature_lists.values() for array in values}
    if not dimensions:
        raise RuntimeError("No feature prototypes were extracted")
    common_dim = min(dimensions)
    if pca_dim > 0:
        common_dim = min(common_dim, pca_dim)
    aligned: dict[str, np.ndarray] = {}
    alignment = {"original_dimensions": sorted(dimensions), "common_dimension": common_dim, "method": "none"}
    for model_id, values in feature_lists.items():
        matrix = np.stack(values, axis=0).astype(np.float32)
        if matrix.shape[1] != common_dim:
            components = min(common_dim, matrix.shape[0], matrix.shape[1])
            if components < 1:
                raise RuntimeError(f"Cannot align feature dimension for {model_id}: {matrix.shape}")
            reducer = PCA(n_components=components, random_state=seed)
            matrix = reducer.fit_transform(matrix).astype(np.float32)
            alignment["method"] = "per_model_pca_to_common_dimension"
            alignment.setdefault("per_model", {})[model_id] = {
                "input_dim": int(values[0].shape[0]),
                "output_dim": int(matrix.shape[1]),
                "explained_variance_ratio_sum": float(reducer.explained_variance_ratio_.sum()),
            }
        aligned[model_id] = matrix
    actual_dim = min(matrix.shape[1] for matrix in aligned.values())
    if any(matrix.shape[1] != actual_dim for matrix in aligned.values()):
        aligned = {model_id: matrix[:, :actual_dim] for model_id, matrix in aligned.items()}
        alignment["common_dimension"] = actual_dim
    # Standardize each vector before the joint embedding, as in the usual
    # prototype-analysis workflow; this prevents feature scale from selecting
    # a model merely because its decoder has a larger activation magnitude.
    concatenated = np.concatenate(list(aligned.values()), axis=0)
    mean = concatenated.mean(axis=0, keepdims=True)
    std = concatenated.std(axis=0, keepdims=True)
    aligned = {model_id: (matrix - mean) / (std + 1e-6) for model_id, matrix in aligned.items()}
    alignment["standardized"] = True
    return aligned, alignment


def embed_and_plot(
    coordinates: np.ndarray,
    records: list[dict],
    output_dir: Path,
    perplexity: float,
    iterations: int,
    seed: int,
) -> dict:
    import matplotlib.pyplot as plt

    count = coordinates.shape[0]
    if count < 3:
        raise RuntimeError("At least three prototypes are needed for t-SNE")
    effective_perplexity = min(float(perplexity), float(max(1, count - 1)))
    if effective_perplexity >= count:
        effective_perplexity = max(1.0, count / 3.0)
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
    }
    try:
        embedder = TSNE(max_iter=iterations, **tsne_kwargs)
    except TypeError:  # scikit-learn < 1.5 used ``n_iter`` instead.
        embedder = TSNE(n_iter=iterations, **tsne_kwargs)
    embedding = embedder.fit_transform(coordinates)
    for record, point in zip(records, embedding):
        record["x"] = float(point[0])
        record["y"] = float(point[1])

    model_ids = list(dict.fromkeys(record["model_id"] for record in records))
    marker_by_model = {model_id: marker for model_id, marker in zip(model_ids, ("o", "^", "s", "D", "P", "X"))}
    colors = {0: "#1f77b4", 1: "#d62728"}

    def plot(source_filter: str | None, path: Path, title: str) -> None:
        selected = [record for record in records if source_filter is None or record["source"] == source_filter]
        figure, axis = plt.subplots(figsize=(8.4, 6.4), dpi=220)
        for model_id in model_ids:
            for class_id in sorted({int(item["class_id"]) for item in selected}):
                points = [item for item in selected if item["model_id"] == model_id and int(item["class_id"]) == class_id]
                if not points:
                    continue
                axis.scatter(
                    [item["x"] for item in points],
                    [item["y"] for item in points],
                    s=26,
                    alpha=0.72,
                    c=colors.get(class_id, "#2ca02c"),
                    marker=marker_by_model[model_id],
                    edgecolors="white",
                    linewidths=0.25,
                    label=f"{model_id} / class {class_id}",
                )
        axis.set_title(title)
        axis.set_xlabel("t-SNE-1")
        axis.set_ylabel("t-SNE-2")
        axis.grid(alpha=0.15)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            # Remove duplicate labels while retaining deterministic ordering.
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), loc="best", fontsize=8, frameon=True)
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)

    plot(None, output_dir / "tsne_all.png", "PMDFNet feature prototypes: GT and Pred")
    plot("gt", output_dir / "tsne_gt.png", "PMDFNet feature prototypes: GT regions")
    plot("pred", output_dir / "tsne_pred.png", "PMDFNet feature prototypes: predicted regions")
    return {"perplexity": effective_perplexity, "iterations": iterations, "points": count}


def main(argv=None) -> int:
    args = parse_args(argv)
    model_ids = parse_id_list(args.model_ids, DEFAULT_MODEL_IDS)
    class_ids = [int(value.strip()) for value in args.class_ids.split(",") if value.strip()]
    if not class_ids:
        raise ValueError("--class-ids must contain at least one class")
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device} but CUDA is unavailable")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    datasets = {}
    metadata = {}
    selected_names: set[str] | None = None
    for model_id in model_ids:
        run, model, resolved_name, config_payload, input_channels = resolve_model(model_id, args, device)
        dataset = MappedLabeledDataset(
            args.data_root,
            args.split,
            input_channels,
            normalization=normalization_for(config_payload, args.split),
            resize_to=resize_for(config_payload, args.split, args.resize),
        )
        indices = selected_indices(dataset, argparse.Namespace(class_ids=class_ids, max_images_per_class=args.max_images_per_class, selection_seed=args.selection_seed))
        names = {dataset.image_files[index].name for index in indices}
        if selected_names is None:
            selected_names = names
        else:
            selected_names &= names
        models[model_id] = model
        datasets[model_id] = dataset
        metadata[model_id] = {
            "run": run,
            "resolved_model_name": resolved_name,
            "input_channels": input_channels,
            "normalization": normalization_for(config_payload, args.split),
            "resize_to": resize_for(config_payload, args.split, args.resize),
        }

    if not selected_names:
        raise RuntimeError("No common selected image IDs across the requested models")

    feature_lists: dict[str, list[np.ndarray]] = {}
    record_lists: dict[str, list[dict]] = {}
    layer_names: dict[str, str] = {}
    for model_id in model_ids:
        model = models[model_id]
        dataset = datasets[model_id]
        name_to_index = {path.name: index for index, path in enumerate(dataset.image_files)}
        model_indices = [name_to_index[name] for name in sorted(selected_names) if name in name_to_index]
        if not model_indices:
            raise RuntimeError(f"No common selected image IDs can be indexed for {model_id}")
        probe = dataset[model_indices[0]][0].unsqueeze(0).to(device)
        layer_name, layer = find_spatial_layer(model, probe, args.feature_layer)
        features, records = model_prototypes(
            model_id,
            model,
            dataset,
            layer_name,
            layer,
            model_indices,
            class_ids,
            device,
        )
        if not features:
            raise RuntimeError(f"No prototypes extracted for {model_id}")
        feature_lists[model_id] = features
        record_lists[model_id] = records
        layer_names[model_id] = layer_name

    if args.max_prototypes > 0:
        rng = np.random.default_rng(args.selection_seed)
        for model_id in model_ids:
            records = record_lists[model_id]
            if len(records) > args.max_prototypes:
                selected = sorted(rng.choice(len(records), size=args.max_prototypes, replace=False).tolist())
                feature_lists[model_id] = [feature_lists[model_id][index] for index in selected]
                record_lists[model_id] = [records[index] for index in selected]

    aligned, alignment = align_features(feature_lists, args.pca_dim, args.selection_seed)
    all_features = np.concatenate([aligned[model_id] for model_id in model_ids], axis=0)
    all_records = [record for model_id in model_ids for record in record_lists[model_id]]
    tsne_info = embed_and_plot(
        all_features,
        all_records,
        output_dir,
        args.perplexity,
        args.tsne_iterations,
        args.selection_seed,
    )

    np.savez_compressed(output_dir / "prototypes_aligned.npz", **{model_id: aligned[model_id] for model_id in model_ids})
    with (output_dir / "prototypes.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for records in record_lists.values() for record in records for key in record})
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for records in record_lists.values():
            writer.writerows(records)
    with (output_dir / "tsne_coordinates.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for record in all_records for key in record})
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    manifest = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "prototype_tsne",
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "split": args.split,
        "model_ids": model_ids,
        "seed": args.seed,
        "device": str(device),
        "class_ids": class_ids,
        "selection": {
            "rule": "fixed_seeded_gt_presence_per_class",
            "selection_seed": args.selection_seed,
            "max_images_per_class": args.max_images_per_class,
            "shared_image_count": len(selected_names),
        },
        "feature_layers": layer_names,
        "models": metadata,
        "alignment": alignment,
        "tsne": tsne_info,
        "outputs": {
            "prototypes": str(output_dir / "prototypes.csv"),
            "coordinates": str(output_dir / "tsne_coordinates.csv"),
            "figure_all": str(output_dir / "tsne_all.png"),
            "figure_gt": str(output_dir / "tsne_gt.png"),
            "figure_pred": str(output_dir / "tsne_pred.png"),
        },
        "interpretation_note": "t-SNE is descriptive and stochastic; use it to visualize feature organization, not as a standalone significance test.",
    }
    save_json(output_dir / "tsne_manifest.json", manifest)
    print(json.dumps({"status": "completed", "models": model_ids, "points": len(all_records), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
