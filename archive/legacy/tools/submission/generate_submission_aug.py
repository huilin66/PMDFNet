import argparse
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

from dataset import MarsSegDatasetInferV0, MarsSegDatasetInferV1
from env_utils import get_data_root
from networks import get_model

def generate_submission(
    model_name,
    checkpoint_path,
    data_root,
    output_zip_path="submission.zip",
    remove=True,
    augs=None,
    temp_dir="temp_submission_masks",
    ana=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    model = get_model(model_name, in_channels=7, num_classes=2).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using 'python train.py'")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint.state_dict())

    model.eval()

    print(f"Loading test data from {data_root}...")
    test_dir = os.path.join(data_root, "test", "images")
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    if ana:
        test_dataset = MarsSegDatasetInferV0(root_dir=data_root, split="test")
        temp_dir = f"ana/{model_name}"
    else:
        test_dataset = MarsSegDatasetInferV1(root_dir=data_root, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Starting inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (images, image_names) in enumerate(
            tqdm(test_loader, desc="Inferencing")
        ):
            images = images.to(device)
            orig_h, orig_w = images.shape[2], images.shape[3]

            final_logits = []
            for scale in augs["ms_scale"]:
                if scale != 1.0:
                    h = int(orig_h * scale + 0.5)
                    w = int(orig_w * scale + 0.5)
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
            pred_mask = torch.argmax(output, dim=1)  # [1, 128, 128]
            pred_mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)

            if i == 0:
                print(
                    f"DEBUG: Sample output values: {np.unique(pred_mask_np)} (Should be [0, 1] for binary)"
                )

            filename = image_names[0]
            save_path = os.path.join(temp_dir, filename)
            tifffile.imwrite(save_path, pred_mask_np)

    output_zip_path = datetime.now().strftime("submission_%Y%m%d_%H%M%S.zip")
    output_zip_path = os.path.join("submission", output_zip_path)
    os.makedirs("submission", exist_ok=True)
    print(f"Creating archive: {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".tif"):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=file)

    if remove and not ana:
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
    print(f"Submission generated successfully: {os.path.abspath(output_zip_path)}")


if __name__ == "__main__":
    pass
    root = get_data_root(
        "/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2"
    )
    outputs = r"/localnvme/project/M3LSNet/outputs"
    result_dict = {
        "segformer_convnexttiny": "20260301_220437",
        "dual_segformer_convnexttiny_chv1_add": "20260301_223101",
    }
    augs = {
        "flip_v": True,
        "flip_h": True,
        "flip_vh": True,
        "rotate_90": True,
        "rotate_180": True,
        "rotate_270": True,
        "ms_scale": [1.0, 1.5],
    }
    for k, v in result_dict.items():
        generate_submission(
            k,
            os.path.join(outputs, v, "checkpoints", "last.pth"),
            root,
            remove=False,
            augs=augs,
            ana=True,
        )
