"""Prototype-level t-SNE analysis for frozen PMDFNet checkpoints.

For each selected image and semantic class, this script computes one mean
feature vector over the GT region and one over the predicted region.  All
requested models are embedded jointly so that their coordinates share one
descriptive reference space.  In addition to joint overview figures, the
script writes one figure per model.  Each model figure shows the selected
foreground/background prototype distributions without mixing the points from
other models into the panel, matching the MRSN paper-style presentation.
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
DEFAULT_CLASS_NAMES = {0: "Background", 1: "Foreground"}
DEFAULT_SOURCES = ("gt",)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract feature prototypes and create joint plus per-model t-SNE plots."
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
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated image filenames; overrides the fixed class-presence selection.",
    )
    parser.add_argument(
        "--sources",
        default="gt",
        help="Sources rendered in each figure: comma-separated gt and/or pred. Default: gt.",
    )
    parser.add_argument(
        "--class-names",
        default="Background,Foreground",
        help="Comma-separated class labels in class-id order; default is Background,Foreground.",
    )
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

    if getattr(args, "ids", "").strip():
        name_to_index = {path.name: index for index, path in enumerate(dataset.image_files)}
        names = [item.strip() for item in args.ids.split(",") if item.strip()]
        missing = [name for name in names if name not in name_to_index]
        if missing:
            raise FileNotFoundError(f"Requested image IDs are missing from {dataset.split}: {missing}")
        return [name_to_index[name] for name in names]

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


def parse_sources(value: str) -> list[str]:
    sources = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(sources) - {"gt", "pred"})
    if invalid:
        raise ValueError(f"--sources contains unsupported values: {invalid}; use gt and/or pred")
    return list(dict.fromkeys(sources)) or list(DEFAULT_SOURCES)


def parse_class_names(value: str, class_ids: list[int]) -> dict[int, str]:
    names = [item.strip() for item in value.split(",")]
    result = dict(DEFAULT_CLASS_NAMES)
    for class_id, name in enumerate(names):
        if name:
            result[class_id] = name
    for class_id in class_ids:
        result.setdefault(class_id, f"class {class_id}")
    return result


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
                    for class_id in class_ids:
                        # Resize the class-specific mask directly.  Using a
                        # resized foreground mask as an additional gate would
                        # silently remove every background prototype.
                        class_mask = resize_mask(mask == class_id, tuple(feature.shape[-2:]))
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
    sources: list[str],
    class_names: dict[int, str],
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
    marker_by_model = {
        model_id: marker
        for model_id, marker in zip(model_ids, ("o", "^", "s", "D", "P", "X", "v", "<", ">"))
    }
    marker_by_source = {"gt": "o", "pred": "^"}
    colors = {
        0: "#4c78a8",
        1: "#e45756",
        2: "#54a24b",
        3: "#f2cf5b",
    }

    def plot(
        path: Path,
        title: str,
        model_filter: str | None = None,
        source_filter: str | None = None,
        include_model_legend: bool = False,
    ) -> None:
        selected = [
            record
            for record in records
            if (model_filter is None or record["model_id"] == model_filter)
            and (source_filter is None or record["source"] == source_filter)
            and record["source"] in sources
        ]
        if not selected:
            return
        figure, axis = plt.subplots(figsize=(8.4, 6.4), dpi=220)
        selected_models = [
            model_id for model_id in model_ids if any(item["model_id"] == model_id for item in selected)
        ]
        selected_sources = [source for source in sources if any(item["source"] == source for item in selected)]
        selected_class_ids = sorted({int(item["class_id"]) for item in selected})
        for model_id in selected_models:
            for source in selected_sources:
                for class_id in selected_class_ids:
                    points = [
                        item
                        for item in selected
                        if item["model_id"] == model_id
                        and item["source"] == source
                        and int(item["class_id"]) == class_id
                    ]
                    if not points:
                        continue
                    class_label = class_names.get(class_id, f"class {class_id}")
                    source_label = source.upper()
                    model_label = model_id if include_model_legend else None
                    label = (
                        f"{model_label} / {source_label} / {class_label}"
                        if model_label
                        else f"{source_label} / {class_label}"
                    )
                    axis.scatter(
                        [item["x"] for item in points],
                        [item["y"] for item in points],
                        s=30,
                        alpha=0.76,
                        c=colors.get(class_id, "#2ca02c"),
                        marker=marker_by_source.get(source) or marker_by_model.get(model_id, "o"),
                        edgecolors="white" if source == "gt" else "#333333",
                        linewidths=0.35,
                        label=label,
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

    output_paths = {"joint": str(output_dir / "tsne_all.png")}
    plot(
        output_dir / "tsne_all.png",
        "PMDFNet feature prototypes: selected models",
        include_model_legend=True,
    )
    for source in ("gt", "pred"):
        if source not in sources:
            continue
        path = output_dir / f"tsne_{source}.png"
        plot(
            path,
            f"PMDFNet feature prototypes: {source.upper()} regions",
            source_filter=source,
            include_model_legend=True,
        )
        output_paths[source] = str(path)

    per_model_dir = output_dir / "by_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    per_model_paths = {}
    for model_id in model_ids:
        path = per_model_dir / f"{safe_name(model_id)}.png"
        plot(
            path,
            f"{model_id} feature prototypes: foreground/background",
            model_filter=model_id,
            include_model_legend=False,
        )
        per_model_paths[model_id] = str(path)

    return {
        "perplexity": effective_perplexity,
        "iterations": iterations,
        "points": count,
        "sources_rendered": sources,
        "joint_figures": output_paths,
        "per_model_figures": per_model_paths,
        "per_model_figure_rule": "one figure per selected model; class colors identify background/foreground and source markers identify GT/Pred",
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    model_ids = parse_id_list(args.model_ids, DEFAULT_MODEL_IDS)
    class_ids = [int(value.strip()) for value in args.class_ids.split(",") if value.strip()]
    if not class_ids:
        raise ValueError("--class-ids must contain at least one class")
    sources = parse_sources(args.sources)
    class_names = parse_class_names(args.class_names, class_ids)
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
        indices = selected_indices(
            dataset,
            argparse.Namespace(
                class_ids=class_ids,
                max_images_per_class=args.max_images_per_class,
                selection_seed=args.selection_seed,
                ids=args.ids,
            ),
        )
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
    plotted_features = []
    all_records = []
    for model_id in model_ids:
        keep = [
            index
            for index, record in enumerate(record_lists[model_id])
            if record["source"] in sources
        ]
        if not keep:
            raise RuntimeError(f"No prototypes for rendered sources {sources} in {model_id}")
        plotted_features.append(aligned[model_id][keep])
        all_records.extend(record_lists[model_id][index] for index in keep)
    all_features = np.concatenate(plotted_features, axis=0)
    tsne_info = embed_and_plot(
        all_features,
        all_records,
        output_dir,
        args.perplexity,
        args.tsne_iterations,
        args.selection_seed,
        sources,
        class_names,
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
        "image_ids": list(
            dict.fromkeys(record["image"] for record in all_records if record["source"] == sources[0])
        ),
        "class_names": {str(class_id): class_names.get(class_id, f"class {class_id}") for class_id in class_ids},
        "sources_extracted": ["gt", "pred"],
        "sources_rendered": sources,
        "selection": {
            "rule": "explicit_image_ids" if args.ids.strip() else "fixed_seeded_gt_presence_per_class",
            "selection_seed": args.selection_seed,
            "max_images_per_class": args.max_images_per_class,
            "shared_image_count": len(selected_names),
            "explicit_ids": [item.strip() for item in args.ids.split(",") if item.strip()],
        },
        "feature_layers": layer_names,
        "models": metadata,
        "alignment": alignment,
        "tsne": tsne_info,
        "outputs": {
            "prototypes": str(output_dir / "prototypes.csv"),
            "coordinates": str(output_dir / "tsne_coordinates.csv"),
            "figures": tsne_info.get("joint_figures", {}),
            "per_model_figures": tsne_info.get("per_model_figures", {}),
        },
        "interpretation_note": "t-SNE is descriptive and stochastic; use it to visualize feature organization, not as a standalone significance test.",
        "figure_note": "The per-model figures are views of one joint embedding, so their axes remain comparable across models. The default rendered source is the GT-defined region, following the MRSN prototype analysis.",
    }
    save_json(output_dir / "tsne_manifest.json", manifest)
    print(json.dumps({"status": "completed", "models": model_ids, "points": len(all_records), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
