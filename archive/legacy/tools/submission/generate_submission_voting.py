import gc
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MarsSegDatasetInferV0, MarsSegDatasetInferV1, MarsSegDatasetInferV2
from env_utils import get_data_root
from networks import get_model

def run_inference_for_model(
    model_name, checkpoint_path, data_root, temp_dir, augs, device, ana=False
):
    print(f"\n{'=' * 50}")
    print(f"🚀 Starting inference for model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'=' * 50}")

    model = get_model(model_name, in_channels=7, num_classes=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint.state_dict())
    model.eval()

    if ana:
        test_dataset = MarsSegDatasetInferV0(root_dir=data_root, split="test")
    elif "1st" in model_name:
        test_dataset = MarsSegDatasetInferV1(root_dir=data_root, split="test")
    elif "2nd" in model_name:
        test_dataset = MarsSegDatasetInferV2(root_dir=data_root, split="test")
    else:
        test_dataset = MarsSegDatasetInferV1(root_dir=data_root, split="test")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model_temp_dir = os.path.join(temp_dir, model_name)
    prob_dir = os.path.join(model_temp_dir, "probs")
    mask_dir = os.path.join(model_temp_dir, "masks")
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, image_names) in enumerate(
            tqdm(test_loader, desc=f"Inferencing {model_name}")
        ):
            images = images.to(device)
            orig_h, orig_w = images.shape[2], images.shape[3]
            filename = image_names[0]

            final_logits = []
            for scale in augs["ms_scale"]:
                if scale != 1.0:
                    h, w = int(orig_h * scale + 0.5), int(orig_w * scale + 0.5)
                    curr_images = F.interpolate(images, (h, w), mode="bilinear")
                else:
                    curr_images = images

                images_aug = [curr_images]

                if augs["flip_v"]:
                    images_aug.append(torch.flip(curr_images, dims=[2]))
                if augs["flip_h"]:
                    images_aug.append(torch.flip(curr_images, dims=[3]))
                if augs["flip_vh"]:
                    images_aug.append(torch.flip(curr_images, dims=[2, 3]))
                if augs["rotate_90"]:
                    images_aug.append(torch.rot90(curr_images, k=1, dims=[2, 3]))
                if augs["rotate_180"]:
                    images_aug.append(torch.rot90(curr_images, k=2, dims=[2, 3]))
                if augs["rotate_270"]:
                    images_aug.append(torch.rot90(curr_images, k=3, dims=[2, 3]))

                images_aug = torch.cat(images_aug, dim=0)
                logit = model(images_aug)

                if scale != 1.0:
                    logit = F.interpolate(logit, (orig_h, orig_w), mode="bilinear")

                logits = [logit[0]]
                pred_idx = 1

                if augs["flip_v"]:
                    logits.append(torch.flip(logit[pred_idx], dims=[1]))
                    pred_idx += 1
                if augs["flip_h"]:
                    logits.append(torch.flip(logit[pred_idx], dims=[2]))
                    pred_idx += 1
                if augs["flip_vh"]:
                    logits.append(torch.flip(logit[pred_idx], dims=[1, 2]))
                    pred_idx += 1
                if augs["rotate_90"]:
                    logits.append(torch.rot90(logit[pred_idx], k=3, dims=[1, 2]))
                    pred_idx += 1
                if augs["rotate_180"]:
                    logits.append(torch.rot90(logit[pred_idx], k=2, dims=[1, 2]))
                    pred_idx += 1
                if augs["rotate_270"]:
                    logits.append(torch.rot90(logit[pred_idx], k=1, dims=[1, 2]))
                    pred_idx += 1

                final_logits.extend(logits)

            output = torch.mean(torch.stack(final_logits, dim=0), dim=0, keepdim=True)

            prob_map = (
                F.softmax(output, dim=1)[0, 1, :, :].cpu().numpy().astype(np.float16)
            )
            np.save(os.path.join(prob_dir, filename.replace(".tif", ".npy")), prob_map)

            pred_mask = (
                torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            )
            tifffile.imwrite(os.path.join(mask_dir, filename), pred_mask)

    del model
    torch.cuda.empty_cache()
    gc.collect()


def generate_ensembles(temp_dir, model_names, output_base, weights=None):
    print(f"\n{'=' * 50}")
    print("Starting Ensemble (Hard Voting & Soft Voting)...")
    print(f"{'=' * 50}")

    hard_vote_dir = os.path.join(temp_dir, "hard_voting_masks")
    soft_vote_dir = os.path.join(temp_dir, "soft_voting_masks")
    os.makedirs(hard_vote_dir, exist_ok=True)
    os.makedirs(soft_vote_dir, exist_ok=True)

    num_models = len(model_names)
    if weights is None:
        weights = [1.0] * num_models
    weights = np.array(weights) / np.sum(weights)

    base_prob_dir = os.path.join(temp_dir, model_names[0], "probs")
    os.makedirs(base_prob_dir, exist_ok=True)
    filenames = [
        f.replace(".npy", ".tif")
        for f in os.listdir(base_prob_dir)
        if f.endswith(".npy")
    ]

    for filename in tqdm(filenames, desc="Ensembling"):
        npy_name = filename.replace(".tif", ".npy")

        masks = []
        probs = []

        for i, model_name in enumerate(model_names):
            mask_path = os.path.join(temp_dir, model_name, "masks", filename)
            masks.append(tifffile.imread(mask_path))

            prob_path = os.path.join(temp_dir, model_name, "probs", npy_name)
            probs.append(np.load(prob_path) * weights[i])

        stacked_masks = np.stack(masks, axis=0)
        sum_mask = np.sum(stacked_masks, axis=0)
        hard_final = (sum_mask >= (num_models / 2.0)).astype(np.uint8)
        tifffile.imwrite(os.path.join(hard_vote_dir, filename), hard_final)

        sum_prob = np.sum(probs, axis=0)
        soft_final = (sum_prob > 0.5).astype(np.uint8)
        tifffile.imwrite(os.path.join(soft_vote_dir, filename), soft_final)

    return hard_vote_dir, soft_vote_dir


def create_zip(source_dir, zip_filename):
    print(f"Creating zip archive: {zip_filename}")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".tif"):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=file)
    print(f"Saved: {zip_filename}")


def main(ana=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = get_data_root(
        "/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2"
    )
    # data_root = "/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_test_data_2nd_phase_updateB2"
    outputs_dir = r"/localnvme/project/M3LSNet/outputs"

    temp_dir = "ensemble_temp_files" if not ana else "./ana/ensemble_temp_files"
    final_output_dir = "submission_ensembles"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    augs = {
        "flip_v": True,
        "flip_h": True,
        "flip_vh": True,
        "rotate_90": True,
        "rotate_180": True,
        "rotate_270": True,
        "ms_scale": [1.0, 1.5],
    }

    result_dict = {
        "dual_segformer_convnexttiny_chv1_add": "20260301_223101",
        "dual_segformer_convnextlarge_chv1_add": "20260302_133033",
    }

    custom_weights = [1.0] * len(result_dict)
    model_names = list(result_dict.keys())

    for model_name, timestamp in result_dict.items():
        checkpoint_path = os.path.join(
            outputs_dir, timestamp, "checkpoints", "last.pth"
        )
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found -> {checkpoint_path}")
            continue

        run_inference_for_model(
            model_name, checkpoint_path, data_root, temp_dir, augs, device, ana=ana
        )

    hard_dir, soft_dir = generate_ensembles(
        temp_dir, model_names, final_output_dir, weights=custom_weights
    )

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    hard_zip = os.path.join(final_output_dir, f"submission_HardVote_{time_str}.zip")
    soft_zip = os.path.join(final_output_dir, f"submission_SoftVote_{time_str}.zip")

    create_zip(hard_dir, hard_zip)
    create_zip(soft_dir, soft_zip)

    print("\nCleaning up temporary files...")
    if not ana:
        shutil.rmtree(temp_dir)
    print("All done! You can now submit the zip files.")


if __name__ == "__main__":
    main()
