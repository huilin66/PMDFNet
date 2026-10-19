import csv
import os

import cv2
import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import dilation, disk
from tqdm import tqdm

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


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


def add_label(img, text, color=(255, 255, 255)):
    h, w = img.shape[:2]
    label_bg = np.zeros((30, w, 3), dtype=np.uint8)
    cv2.putText(
        label_bg, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA
    )
    return cv2.vconcat([label_bg, img])


def to_bgr(img_gray):
    if img_gray.ndim == 2:
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    return img_gray


def normalize_to_uint8(data):
    data = data.astype(np.float32)
    min_val = np.min(data)
    max_val = np.max(data)

    if max_val - min_val < 1e-5:
        return np.zeros_like(data, dtype=np.uint8)

    norm_data = (data - min_val) / (max_val - min_val) * 255.0
    return norm_data.astype(np.uint8)


def add_label_header(img, text, bg_color=(0, 0, 0), txt_color=(255, 255, 255)):
    h, w, c = img.shape
    header_h = 30
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    header[:] = bg_color

    font_scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    text_x = (w - text_w) // 2
    text_y = (header_h + text_h) // 2

    cv2.putText(
        header,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        txt_color,
        thickness,
        cv2.LINE_AA,
    )

    return cv2.vconcat([header, img])


def generate_comprehensive_vis(img_dir, vis_dir, gt_dir, pred_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    gt_files = [f for f in os.listdir(gt_dir) if f.endswith((".png", ".tif", ".tiff"))]

    if not gt_files:
        print("GT empty")
        return

    def get_file_map(directory):
        mapping = {}
        for f in os.listdir(directory):
            name = os.path.splitext(f)[0]
            mapping[name] = os.path.join(directory, f)
        return mapping

    img_map = get_file_map(img_dir)
    pred_map = get_file_map(pred_dir)

    for gt_file in tqdm(gt_files, desc="Generating VIS", unit="img"):
        base_name = os.path.splitext(gt_file)[0]

        if base_name not in img_map or base_name not in pred_map:
            continue

        gt_path = os.path.join(gt_dir, gt_file)
        img_path = img_map[base_name]
        pred_path = pred_map[base_name]
        vis_path = os.path.join(vis_dir, base_name + ".jpg")
        save_path = os.path.join(output_dir, base_name + "_12grid.jpg")

        try:
            multi_band_img = tiff.imread(img_path)
        except Exception as e:
            print(f"读取TIFF失败 {base_name}: {e}")
            continue

        if multi_band_img.ndim == 3 and multi_band_img.shape[0] == 7:
            multi_band_img = np.transpose(multi_band_img, (1, 2, 0))

        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        vis_polyon_img = cv2.imread(vis_path, cv2.IMREAD_COLOR)

        h, w = gt_img.shape
        gt_bin = gt_img
        pred_bin = pred_img

        vis_images = []

        titles_row1 = ["B1: Thermal", "B2: Slope", "B3: DEM", "B4: Gray(CTX)"]
        for i in range(4):
            band_data = multi_band_img[:, :, i]
            norm_band = normalize_to_uint8(band_data)
            vis_img = to_bgr(norm_band)
            vis_images.append(add_label_header(vis_img, titles_row1[i]))

        b_r = normalize_to_uint8(multi_band_img[:, :, 4])
        b_g = normalize_to_uint8(multi_band_img[:, :, 5])
        b_b = normalize_to_uint8(multi_band_img[:, :, 6])

        titles_row2 = ["B5: Red", "B6: Green", "B7: Blue", "RGB Composite"]

        vis_images.append(add_label_header(to_bgr(b_r), titles_row2[0]))
        vis_images.append(add_label_header(to_bgr(b_g), titles_row2[1]))
        vis_images.append(add_label_header(to_bgr(b_b), titles_row2[2]))

        rgb_composite = cv2.merge([b_b, b_g, b_r])
        vis_images.append(add_label_header(rgb_composite, titles_row2[3]))

        vis_gt = to_bgr(gt_bin * 255)
        vis_images.append(add_label_header(vis_gt, "GT"))

        vis_pred = to_bgr(pred_bin * 255)
        vis_images.append(add_label_header(vis_pred, "Pred Mask"))

        status_map = 2 * gt_bin + pred_bin
        error_vis = np.zeros((h, w, 3), dtype=np.uint8)
        error_vis[status_map == 1] = [0, 0, 255]
        error_vis[status_map == 2] = [255, 0, 0]
        error_vis[status_map == 3] = [255, 255, 255]
        vis_images.append(add_label_header(error_vis, "Error Map"))

        vis_images.append(add_label_header(vis_polyon_img, "Vis"))

        row1 = cv2.hconcat(vis_images[0:4])
        row2 = cv2.hconcat(vis_images[4:8])
        row3 = cv2.hconcat(vis_images[8:12])
        final_grid = cv2.vconcat([row1, row2, row3])

        cv2.imwrite(save_path, final_grid)


def calculate_metrics(gt, pred):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    epsilon = 1e-6
    iou = tp / (tp + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)

    return float(iou), float(dice)


# def add_label_header_pil(img, label, height=35):
#     return img


def add_label_header_pil(img, label, height=35):
    h, w = img.shape[:2]
    header = np.full((height, w, 3), 255, dtype=np.uint8)
    pil_header = Image.fromarray(header)
    draw = ImageDraw.Draw(pil_header)

    font_size = 20
    try:
        font = ImageFont.truetype("times.ttf", size=font_size)
    except IOError:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
                size=font_size,
            )
        except IOError:
            try:
                font = ImageFont.truetype("LiberationSerif-Regular.ttf", size=font_size)
            except IOError:
                font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = (w - text_w) // 2
    text_y = (height - text_h) // 2 - 2

    draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0))
    header_cv2 = np.array(pil_header)
    return cv2.vconcat([header_cv2, img])


def generate_models_comprehensive_vis(img_dir, gt_dir, pred_dirs_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    gt_files = [f for f in os.listdir(gt_dir) if f.endswith((".png", ".tif", ".tiff"))]
    if not gt_files:
        print("GT empty")
        return

    def get_file_map(directory):
        if not os.path.exists(directory):
            return {}
        return {
            os.path.splitext(f)[0]: os.path.join(directory, f)
            for f in os.listdir(directory)
        }

    img_map = get_file_map(img_dir)

    models_pred_maps = {
        model_name: get_file_map(p_dir) for model_name, p_dir in pred_dirs_dict.items()
    }
    csv_data = []

    model_names = list(pred_dirs_dict.keys())
    fieldnames = ["Image"] + model_names

    for gt_file in tqdm(gt_files, desc="Generating VIS", unit="img"):
        base_name = os.path.splitext(gt_file)[0]

        if base_name not in img_map:
            continue

        gt_path = os.path.join(gt_dir, gt_file)
        img_path = img_map[base_name]
        save_path = os.path.join(output_dir, base_name + "_compare.png")

        try:
            multi_band_img = tiff.imread(img_path)
        except Exception as e:
            print(f"fail to read {base_name}: {e}")
            continue

        if multi_band_img.ndim == 3 and multi_band_img.shape[0] == 7:
            multi_band_img = np.transpose(multi_band_img, (1, 2, 0))

        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt_img.shape

        vis_columns = []

        b_r = normalize_to_uint8(multi_band_img[:, :, 4])
        b_g = normalize_to_uint8(multi_band_img[:, :, 5])
        b_b = normalize_to_uint8(multi_band_img[:, :, 6])
        rgb_composite = cv2.merge([b_b, b_g, b_r])  # OpenCV uses BGR
        vis_columns.append(add_label_header_pil(rgb_composite, "RGB"))

        bands = [(0, "Thermal"), (1, "Slope"), (2, "DEM")]
        for b_idx, title in bands:
            band_data = multi_band_img[:, :, b_idx]
            norm_band = normalize_to_uint8(band_data)
            color_band = cv2.applyColorMap(norm_band, cv2.COLORMAP_JET)
            vis_columns.append(add_label_header_pil(color_band, title))

        vis_gt = to_bgr(gt_img)
        vis_gt *= 255
        vis_columns.append(add_label_header_pil(vis_gt, "GT"))

        row_data = {"Image": base_name}

        for model_name, pred_map in models_pred_maps.items():
            if base_name in pred_map:
                pred_img = cv2.imread(pred_map[base_name], cv2.IMREAD_GRAYSCALE)

                iou, _ = calculate_metrics(gt_img, pred_img)

                vis_pred = to_bgr(pred_img)
                vis_pred *= 255
            else:
                vis_pred = np.zeros((h, w, 3), dtype=np.uint8)
                iou = 0.0

            row_data[model_name] = round(iou, 4)

            title_with_metric = f"{model_name}"
            vis_columns.append(add_label_header_pil(vis_pred, title_with_metric))

        csv_data.append(row_data)

        final_row = cv2.hconcat(vis_columns)
        cv2.imwrite(save_path, final_row)

    # 保存宽表 CSV
    csv_path = os.path.join(output_dir, "evaluation_metrics_iou.csv")
    if csv_data:
        with open(csv_path, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(csv_data)
        print(f"\n✅ All IoU metrics saved to: {csv_path}")


if __name__ == "__main__":
    pass

    generate_models_comprehensive_vis(
        img_dir="/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2/val/images",
        gt_dir="/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2/val/masks",
        pred_dirs_dict={
            "SegFormer": "ana/segformer_convnexttiny",
            "PMDFNet": "ana/dual_segformer_convnexttiny_chv1_add",
            "Ensemble": "ana/ensemble_temp_files/soft_voting_masks",
        },
        output_dir="./vis_models",
    )
