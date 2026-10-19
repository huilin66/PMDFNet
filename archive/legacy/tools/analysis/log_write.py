import os
import re

import pandas as pd


def parse_training_log_to_csv(log_path, output_csv, default_model_name="m3lsnet"):
    """
    Read training log file, extract metrics and write into CSV.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    model_match = re.search(r"Model:\s+([\w\d_]+)", content)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = default_model_name

    folder_match = re.search(
        r"Outputs will be saved to: outputs[/\\]([\w\d_]+)", content
    )
    folder_name = folder_match.group(1) if folder_match else "Unknown"

    normalized_content = content.replace("\n", " ")

    pattern = r"mIoU:\s*([\d\.]+).*?F1:\s*([\d\.]+).*?IoU \(FG\):\s*([\d\.]+).*?IoU \(BG\):\s*([\d\.]+)"
    matches = re.findall(pattern, normalized_content)

    data = []
    for m in matches:
        data.append(
            {
                "mIoU": float(m[0]),
                "F1": float(m[1]),
                "IoU (FG)": float(m[2]),
                "IoU (BG)": float(m[3]),
            }
        )

    if not data:
        print(f"No metrics found in {log_path}")
        data.append({"mIoU": 0, "F1": 0, "IoU (FG)": 0, "IoU (BG)": 0})

    df_metrics = pd.DataFrame(data)

    best_row = df_metrics.loc[df_metrics["mIoU"].idxmax()]

    result = {
        "ID": "",
        "Folder Name": folder_name,
        "Submission Name": "",
        "Best mIoU": best_row["mIoU"],
        "F1": best_row["F1"],
        "IoU (FG)": best_row["IoU (FG)"],
        "IoU (BG)": best_row["IoU (BG)"],
        "online score": "",
        "Model": model_name,
        "Mark": "",
        "Take strategy": "",
    }

    df_result = pd.DataFrame([result])

    if not os.path.exists(output_csv):
        df_result.to_csv(output_csv, index=False)
    else:
        df_result.to_csv(output_csv, mode="a", header=False, index=False)

    print(f"Successfully processed {log_path} -> Best mIoU: {best_row['mIoU']}")


def batch_write(logs_dir, output_csv, task_list=None):
    if task_list is None:
        task_list = os.listdir(logs_dir)
    task_list.sort()
    for task_name in task_list:
        log_dir = os.path.join(logs_dir, task_name, "logs")
        log_path = os.path.join(log_dir, f"train_log_{task_name}.txt")
        parse_training_log_to_csv(log_path, output_csv)


if __name__ == "__main__":
    log_dir = "/localnvme/project/dual_segformer_convnext/outputs"
    output_csv = "val_metrics.csv"
    task_list = [
        "20260305_203049",
        "20260305_204800",
        "20260305_212159",
        "20260305_214043",
        "20260305_215720",
        "20260305_221337",
        "20260305_223005",
        "20260305_224737",
        "20260305_230846",
        "20260305_232711",
    ]
    batch_write(log_dir, output_csv, task_list)
