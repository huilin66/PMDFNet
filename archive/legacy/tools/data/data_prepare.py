import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
from skimage.morphology import dilation, disk
from tqdm import tqdm

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
SUBSETS = ["train", "val", "test"]
IMG_DIR_NAME = "images"
MASK_DIR_NAME = "masks"
ANOMALY_THRESHOLD = -1e30
TARGET_BAND_IDX = 1


def normalize(img):
    img = np.nan_to_num(img)
    if img.max() - img.min() == 0:
        return img
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


def get_mask_edges(mask):
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
    return edges > 0


def calculate_edge_overlap(band_img, mask_edges):
    v = np.median(band_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    band_edges = cv2.Canny(band_img, lower, upper)
    band_edges_bool = band_edges > 0

    mask_edges_dilated = dilation(mask_edges, disk(1))
    intersection = np.logical_and(band_edges_bool, mask_edges_dilated)
    if np.sum(mask_edges) == 0:
        return 0.0
    score = np.sum(intersection) / np.sum(mask_edges)
    return score


def visualize_band2_and_edges(img_dir, output_dir, num_samples=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_files = [f for f in os.listdir(img_dir) if f.endswith((".tif", ".tiff"))]

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        save_path = os.path.join(output_dir, f"vis_{img_file}.jpg")
        try:
            img = tiff.imread(img_path)
            if img.ndim == 3 and img.shape[0] < img.shape[2]:
                img = np.transpose(img, (1, 2, 0))

            if img.ndim == 3 and img.shape[2] >= 2:
                band2 = img[:, :, 1]
            else:
                band2 = img
                print(f"Warning: {img_file} is not mutiband image, use first channel")

            invalid_mask = np.isinf(band2) | np.isnan(band2) | (band2 < -1e6)
            valid_data = band2[~invalid_mask]
            if len(valid_data) == 0:
                print(f"Skipping {img_file}: No valid data in Band 2")
                continue

            vmin, vmax = np.percentile(valid_data, 1), np.percentile(valid_data, 99)

            plot_data = band2.copy()
            plot_data[invalid_mask] = np.nan

            norm_img = np.clip(
                (band2 - np.min(valid_data))
                / (np.max(valid_data) - np.min(valid_data))
                * 255,
                0,
                255,
            )
            norm_img = np.nan_to_num(norm_img).astype(np.uint8)

            v = np.median(norm_img)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(norm_img, lower, upper)

            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

            fig = plt.figure(figsize=(15, 5))
            ax1 = plt.subplot(1, 3, 1)
            cmap = plt.cm.magma
            cmap.set_bad("gray", 1.0)
            im = ax1.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            ax1.set_title(f"Band 2: Slope\n(Range: {vmin:.1f} - {vmax:.1f})")
            ax1.axis("off")

            ax2 = plt.subplot(1, 3, 2)
            ax2.imshow(edges, cmap="gray")
            ax2.set_title("Generated Edge Band\n(Based on Slope)")
            ax2.axis("off")

            ax3 = plt.subplot(1, 3, 3)
            ax3.hist(valid_data.flatten(), bins=50, color="orange", alpha=0.7)
            ax3.set_title("Pixel Value Distribution\n(Valid Pixels Only)")
            ax3.set_xlabel("Slope Value")
            ax3.set_ylabel("Count")

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        except Exception as e:
            print(f"Error processing {img_file}: {e}")


def visualize_anomaly_location(img_dir, output_dir, max_samples=10):
    """
    visualize -3.4028235e+38 distribution
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_files = [f for f in os.listdir(img_dir) if f.endswith((".tif", ".tiff"))]

    ANOMALY_THRESHOLD = -1e30

    count = 0
    print(f"scanning (-3.4028235e+38) distribution...")

    for img_file in img_files:
        if count >= max_samples:
            break
        img_path = os.path.join(img_dir, img_file)
        try:
            img = tiff.imread(img_path)
            if img.ndim == 3 and img.shape[0] < img.shape[2]:
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3:
                band2 = img[:, :, 1]
            else:
                band2 = img
            if np.min(band2) > ANOMALY_THRESHOLD:
                continue

            count += 1
            print(f"find anomaly: {img_file}")
            anomaly_mask = band2 < ANOMALY_THRESHOLD
            valid_data = band2.copy()
            valid_min = np.min(band2[~anomaly_mask]) if np.any(~anomaly_mask) else 0
            valid_max = np.max(band2[~anomaly_mask]) if np.any(~anomaly_mask) else 1
            valid_data[anomaly_mask] = valid_min
            norm_bg = (valid_data - valid_min) / (valid_max - valid_min + 1e-6)
            norm_bg = np.clip(norm_bg * 255, 0, 255).astype(np.uint8)
            norm_bg = cv2.cvtColor(norm_bg, cv2.COLOR_GRAY2RGB)  # 转为RGB以便叠加红色

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(norm_bg)
            axes[0].set_title(
                f"Terrain Context\n(Anomaly replaced by min: {valid_min:.1f})"
            )
            axes[0].axis("off")

            axes[1].imshow(anomaly_mask, cmap="gray")
            axes[1].set_title("Anomaly Mask\n(White = -3.4e38 / NoData)")
            axes[1].axis("off")

            overlay = norm_bg.copy()
            overlay[anomaly_mask] = [255, 0, 0]

            axes[2].imshow(overlay)
            axes[2].set_title(
                f"Red Overlay\n(Red = Bad Areas), Total Bad Pixels: {np.sum(anomaly_mask)}"
            )
            axes[2].axis("off")

            plt.suptitle(f"File: {img_file}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"anomaly_check_{img_file}.jpg"))
            plt.close()

        except Exception as e:
            print(f"Error: {e}")

    print(f"find {count} anomaly images")


def fix_layer_by_copying(layer_2d, anomaly_mask_2d):
    """
    fix anomaly by copying neighbor values
    """
    working_layer = layer_2d.astype(np.float32)
    working_layer[anomaly_mask_2d] = np.nan

    df = pd.DataFrame(working_layer)
    df = df.ffill(axis=1).bfill(axis=1)
    df = df.ffill(axis=0).bfill(axis=0)
    return df.values


def get_mask_path(img_path, subset):
    mask_name = img_path.name
    return img_path.parent.parent / MASK_DIR_NAME / mask_name


def fix_nodata(SRC_ROOT, DST_ROOT, SUBSETS=SUBSETS):
    Path(DST_ROOT).mkdir(parents=True, exist_ok=True)
    total_imgs = 0
    total_masks = 0
    for subset in SUBSETS:
        src_img_dir = Path(SRC_ROOT) / subset / IMG_DIR_NAME
        dst_img_dir = Path(DST_ROOT) / subset / IMG_DIR_NAME
        dst_mask_dir = Path(DST_ROOT) / subset / MASK_DIR_NAME
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        if subset != "test":
            dst_mask_dir.mkdir(parents=True, exist_ok=True)
        if not src_img_dir.exists():
            print(f"skip empty {subset}")
            continue

        files = list(src_img_dir.glob("*.tif")) + list(src_img_dir.glob("*.tiff"))
        print(f"fix {subset} with {len(files)} files")

        for img_path in tqdm(files):
            img_data = tiff.imread(str(img_path))
            original_dtype = img_data.dtype
            if img_data.ndim == 3 and img_data.shape[0] < img_data.shape[2]:
                band_idx = TARGET_BAND_IDX
                is_channel_first = True
                target_band = img_data[band_idx, :, :]
            elif img_data.ndim == 3:
                band_idx = TARGET_BAND_IDX
                is_channel_first = False
                target_band = img_data[:, :, band_idx]
            else:
                continue
            anomaly_mask = target_band < ANOMALY_THRESHOLD
            has_anomaly = np.any(anomaly_mask)

            if has_anomaly:
                fixed_band = fix_layer_by_copying(target_band, anomaly_mask)
                if is_channel_first:
                    img_data[band_idx, :, :] = fixed_band.astype(original_dtype)
                else:
                    img_data[:, :, band_idx] = fixed_band.astype(original_dtype)

            tiff.imwrite(
                str(dst_img_dir / img_path.name), img_data, photometric="minisblack"
            )
            total_imgs += 1
            if subset == "test":
                continue

            mask_path = get_mask_path(img_path, subset)
            if mask_path.exists():
                mask_data = tiff.imread(str(mask_path))
                mask_dtype = mask_data.dtype

                if has_anomaly:
                    if mask_data.ndim == 2:
                        fixed_mask = fix_layer_by_copying(mask_data, anomaly_mask)
                        mask_data = fixed_mask.astype(mask_dtype)
                    elif mask_data.ndim == 3:
                        for c in range(mask_data.shape[0]):
                            fixed_layer = fix_layer_by_copying(
                                mask_data[c], anomaly_mask
                            )
                            mask_data[c] = fixed_layer.astype(mask_dtype)

                tiff.imwrite(
                    str(dst_mask_dir / mask_path.name),
                    mask_data,
                    photometric="minisblack",
                    compression="zlib",
                )
                total_masks += 1

    print(f"fixed images save to {DST_ROOT}")
    print(f"update {total_imgs} images, {total_masks} images")


def fix_nodata_test(src_img_dir, dst_img_dir):
    total_imgs = 0
    total_masks = 0

    src_img_dir = Path(src_img_dir)
    dst_img_dir = Path(dst_img_dir)
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    files = list(src_img_dir.glob("*.tif")) + list(src_img_dir.glob("*.tiff"))
    print(f"fix {src_img_dir.name} with {len(files)} files")

    for img_path in tqdm(files):
        img_data = tiff.imread(str(img_path))
        original_dtype = img_data.dtype
        if img_data.ndim == 3 and img_data.shape[0] < img_data.shape[2]:
            band_idx = TARGET_BAND_IDX
            is_channel_first = True
            target_band = img_data[band_idx, :, :]
        elif img_data.ndim == 3:
            band_idx = TARGET_BAND_IDX
            is_channel_first = False
            target_band = img_data[:, :, band_idx]
        else:
            continue

        anomaly_mask = target_band < ANOMALY_THRESHOLD
        has_anomaly = np.any(anomaly_mask)

        if has_anomaly:
            fixed_band = fix_layer_by_copying(target_band, anomaly_mask)
            if is_channel_first:
                img_data[band_idx, :, :] = fixed_band.astype(original_dtype)
            else:
                img_data[:, :, band_idx] = fixed_band.astype(original_dtype)
        tiff.imwrite(
            str(dst_img_dir / img_path.name), img_data, photometric="minisblack"
        )
        total_imgs += 1

    print(f"fixed images save to {dst_img_dir}")
    print(f"update {total_imgs} images, {total_masks} images")


def sta_ms(folder_list):
    """
    statistics of mean std
    """
    if isinstance(folder_list, str):
        folder_list = [folder_list]

    all_img_paths = []
    print("Scanning folders...")
    for folder in folder_list:
        if not os.path.exists(folder):
            print(f"Warning: Folder not found: {folder}")
            continue

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith((".tif", ".tiff"))
        ]
        all_img_paths.extend(files)
        print(f"  - Found {len(files)} images in: {folder}")

    if not all_img_paths:
        print("Error: No images found in any provided folders.")
        return

    print(f"Total images to process: {len(all_img_paths)}")

    try:
        first_img = tiff.imread(all_img_paths[0])
    except Exception as e:
        print(f"Error reading first image: {e}")
        return
    if (
        first_img.ndim == 3
        and first_img.shape[0] < first_img.shape[1]
        and first_img.shape[0] < first_img.shape[2]
    ):
        channels = first_img.shape[0]
        is_channel_first = True
    elif first_img.ndim == 3:
        channels = first_img.shape[2]
        is_channel_first = False
    else:
        channels = 1
        is_channel_first = False

    print(f"Detected {channels} channels. Starting calculation...")

    channel_sum = np.zeros(channels, dtype=np.float64)
    channel_sq_sum = np.zeros(channels, dtype=np.float64)
    channel_pixel_count = np.zeros(channels, dtype=np.float64)

    for img_path in tqdm(all_img_paths, unit="img"):
        try:
            img = tiff.imread(img_path).astype(np.float32)
            img[np.isinf(img)] = np.nan
            img[img < -1e6] = np.nan

            if is_channel_first:
                img_flat = img.reshape(channels, -1).T
            else:
                if channels > 1:
                    img_flat = img.reshape(-1, channels)
                else:
                    img_flat = img.reshape(-1, 1)

            valid_mask = ~np.isnan(img_flat)
            img_safe = np.nan_to_num(img_flat)
            channel_sum += np.sum(img_safe, axis=0)
            channel_sq_sum += np.sum(img_safe**2, axis=0)
            channel_pixel_count += np.sum(valid_mask, axis=0)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    channel_pixel_count[channel_pixel_count == 0] = 1.0

    mean = channel_sum / channel_pixel_count
    std = np.sqrt((channel_sq_sum / channel_pixel_count) - (mean**2))

    print("\n" + "=" * 50)
    print(f"Statistics over {len(folder_list)} folders")
    print("=" * 50)
    print(f"Total Valid Pixels Processed (Band 1): {int(channel_pixel_count[0])}")
    print("-" * 20)

    print("Copy-Paste Format:")
    print(f"mean = {mean.tolist()}")
    print(f"std  = {std.tolist()}")

    print("-" * 20)
    print("Detailed:")
    for i in range(channels):
        print(f"Band {i + 1}: Mean={mean[i]:.4f}, Std={std[i]:.4f}")


def analyze_band_importance(img_dir, mask_dir):
    img_files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
    band_names = [
        "Band 1: Thermal Inertia",
        "Band 2: Slope",
        "Band 3: DEM",
        "Band 4: Grayscale",
        "Band 5: Red",
        "Band 6: Green",
        "Band 7: Blue",
    ]

    results = {
        name: {"edge_score": [], "correlation": [], "mi": []} for name in band_names
    }

    print(f"analyze {len(img_files)} images...")

    for img_file in tqdm(img_files):
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)

        if not os.path.exists(mask_path):
            continue

        try:
            img_data = tiff.imread(img_path)
            if img_data.shape[0] == 7:
                img_data = np.transpose(img_data, (1, 2, 0))

            mask_data = tiff.imread(mask_path)
            if mask_data.ndim == 3:
                mask_data = mask_data[:, :, 0]
            mask_data = (mask_data > 0).astype(np.uint8)

            has_landslide = np.sum(mask_data) > 0
            mask_edges = get_mask_edges(mask_data) if has_landslide else None

            mask_flat = mask_data.flatten()

            for i, name in enumerate(band_names):
                band = img_data[:, :, i]
                band_norm = normalize(band)
                band_flat = band.flatten()

                if has_landslide:
                    edge_score = calculate_edge_overlap(band_norm, mask_edges)
                    results[name]["edge_score"].append(edge_score)

                corr = np.corrcoef(band_flat, mask_flat)[0, 1]
                if not np.isnan(corr):
                    results[name]["correlation"].append(abs(corr))
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    summary = []
    for name in band_names:
        avg_edge = (
            np.mean(results[name]["edge_score"]) if results[name]["edge_score"] else 0
        )
        avg_corr = (
            np.mean(results[name]["correlation"]) if results[name]["correlation"] else 0
        )
        summary.append(
            {
                "Band": name,
                "Edge Alignment Score": avg_edge,
                "Pixel Correlation": avg_corr,
            }
        )

    df = pd.DataFrame(summary)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(data=df, y="Band", x="Edge Alignment Score", palette="viridis")
    plt.title("Method 1: Edge Alignment (Boundary Match)")
    plt.xlabel("Average Overlap Score")

    plt.subplot(1, 2, 2)
    sns.barplot(data=df, y="Band", x="Pixel Correlation", palette="magma")
    plt.title("Method 2: Pixel Intensity Correlation")
    plt.xlabel("Absolute Correlation Coefficient")

    plt.tight_layout()
    plt.show()

    print("\nanalyze results:")
    print(df)

    best_edge = df.loc[df["Edge Alignment Score"].idxmax()]
    print(f"\nbest match: {best_edge['Band']}")


if __name__ == "__main__":
    pass

    # region train data prepare
    # # TODO: update path
    # TRAIN_SRC = '/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase'
    TRAIN_REPAIR = (
        "/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2"
    )
    DATA_ROOT = TRAIN_REPAIR

    # fix_nodata(TRAIN_SRC, TRAIN_REPAIR)

    # sta_ms(os.path.join(DATA_ROOT, "train/images"))
    # sta_ms(os.path.join(DATA_ROOT, "val/images"))
    # sta_ms(os.path.join(DATA_ROOT, "test/images"))

    analyze_band_importance(
        os.path.join(DATA_ROOT, "train/images"), os.path.join(DATA_ROOT, "train/masks")
    )

    # endregion

    # region test data prepare
    # # TODO: update path
    # TEST_SRC = "/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_test_data_2nd_phase"
    # TEST_REPAIR = "/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_test_data_2nd_phase_updateB2/test/images"

    # # fix nodata and organize test images for inference
    # fix_nodata_test(TEST_SRC, TEST_REPAIR)

    # endregion
