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
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MarsSegDatasetInferV1
from networks import get_model

def generate_submission(
    model_name,
    checkpoint_path,
    data_root,
    output_zip_path="submission.zip",
    remove=True,
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

    test_dataset = MarsSegDatasetInferV1(root_dir=data_root, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    temp_dir = "temp_submission_masks"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Starting inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (images, image_names) in enumerate(
            tqdm(test_loader, desc="Inferencing")
        ):
            images = images.to(device)

            output = model(images)
            pred_mask = torch.argmax(output, dim=1)

            pred_mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)

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

    if remove:
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
    print(
        f"[{checkpoint_path.split('/')[-3]}] {model_name}: {os.path.basename(output_zip_path)}"
    )


if __name__ == "__main__":
    pass
    # root =""
    # outputs = r""
    # result_dict = {
    # }
    # for k, v in result_dict.items():
    #     generate_submission(
    #         k, os.path.join(outputs, v, "checkpoints", "last.pth"), root, remove=False
    #     )
