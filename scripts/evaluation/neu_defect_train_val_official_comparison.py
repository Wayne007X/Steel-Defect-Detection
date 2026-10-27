# -*- coding: utf-8 -*-
"""
NEU-DET：clean 与 mosaic000 的 train/val 同口径官方指标对照
只执行 model.val()，不调用 train()，不会更新权重。

建议在 E:\yolo_project 下运行：
    python -u .\neu_defect_train_val_official_comparison.py 2>&1 |
        Tee-Object -FilePath .\train_val_official_comparison_console.txt

注意：
- 训练集指标仅用于诊断，不是独立测试结果，也不能单独证明过拟合。
- 官方 P/R 为 Ultralytics 验证流程计算的汇总指标，不是部署固定阈值下的 P/R。
- train 与 val 使用相同 imgsz、batch、conf、NMS IoU、max_det 等配置。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


# ======================== 路径配置 ========================
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")

WEIGHTS = {
    "neu_defect_clean": Path(
        r"E:\yolo_project\runs\neu_defect_clean\weights\best.pt"
    ),
    "neu_defect_mosaic000": Path(
        r"E:\yolo_project\runs\neu_defect_mosaic000\weights\best.pt"
    ),
}

OUTPUT_BASE = Path(
    r"E:\yolo_project\runs\neu_defect_train_val_official_comparison"
)

# ======================== 统一评估配置 ========================
IMGSZ = 640
BATCH = 2
CONF = 0.001
NMS_IOU = 0.70
MAX_DET = 300
WORKERS = 0
SPLITS = ("train", "val")
RECT = True
AUGMENT = False
PLOTS = False
SAVE_JSON = False
DEVICE = 0 if torch.cuda.is_available() else "cpu"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".webp", ".dng", ".mpo", ".avif", ".heic", ".heif", ".jp2"
}


def normalize_names(raw_names):
    if isinstance(raw_names, list):
        names = {i: str(name) for i, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(k): str(v) for k, v in raw_names.items()}
    else:
        raise ValueError(
            "data.yaml 中的 names 不是列表或字典，无法安全生成类别明细。"
        )

    expected = {
        0: "crazing",
        1: "inclusion",
        2: "patches",
        3: "pitted_surface",
        4: "rolled-in_scale",
        5: "scratches",
    }
    if names != expected:
        raise ValueError(
            "data.yaml 的类别定义与当前项目预期不一致。\n"
            f"实际：{names}\n预期：{expected}\n"
            "请先核对数据配置，避免把类别指标写错。"
        )
    return names


def resolve_split_path(data_cfg, split, yaml_path):
    split_value = data_cfg.get(split)
    if not split_value:
        raise ValueError(f"data.yaml 中缺少 {split!r} 字段。")

    values = split_value if isinstance(split_value, list) else [split_value]
    resolved = []

    data_root = data_cfg.get("path")
    if data_root:
        root = Path(str(data_root))
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
    else:
        root = yaml_path.parent.resolve()

    for value in values:
        p = Path(str(value))
        if not p.is_absolute():
            p = (root / p).resolve()
        resolved.append(p)
    return resolved


def count_images(paths):
    count = 0
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            count += 1
        elif path.is_dir():
            count += sum(
                1 for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
    return count


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metrics(metrics, model_name, split, names):
    box = metrics.box

    # 汇总指标：与 Ultralytics 官方验证日志中的 all 行一致
    overall = {
        "model": model_name,
        "split": split,
        "class_id": "",
        "class_name": "all",
        "imgsz": IMGSZ,
        "batch": BATCH,
        "conf_floor": CONF,
        "nms_iou": NMS_IOU,
        "max_det": MAX_DET,
        "precision": as_float(box.mp),
        "recall": as_float(box.mr),
        "mAP50": as_float(box.map50),
        "mAP50_95": as_float(box.map),
    }

    class_rows = []
    p_arr = list(box.p)
    r_arr = list(box.r)
    ap50_arr = list(box.ap50)
    ap_arr = list(box.ap)
    class_indices = [int(x) for x in list(box.ap_class_index)]
    index_by_class = {class_id: i for i, class_id in enumerate(class_indices)}

    for class_id in sorted(names):
        j = index_by_class.get(class_id)
        row = {
            "model": model_name,
            "split": split,
            "class_id": class_id,
            "class_name": names[class_id],
            "imgsz": IMGSZ,
            "batch": BATCH,
            "conf_floor": CONF,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50_95": None,
        }
        if j is not None:
            row["precision"] = as_float(p_arr[j])
            row["recall"] = as_float(r_arr[j])
            row["mAP50"] = as_float(ap50_arr[j])
            row["mAP50_95"] = as_float(ap_arr[j])
        class_rows.append(row)

    return overall, class_rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"没有数据，无法写入：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"数据 YAML 不存在：{DATA_YAML}")

    for model_name, weight in WEIGHTS.items():
        if not weight.is_file():
            raise FileNotFoundError(f"{model_name} 权重不存在：{weight}")

    data_cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8-sig"))
    names = normalize_names(data_cfg.get("names"))

    print("=" * 78, flush=True)
    print("NEU-DET clean vs mosaic000：train/val 官方验证对照", flush=True)
    print(f"数据 YAML：{DATA_YAML}", flush=True)
    print(f"设备：{DEVICE}", flush=True)
    print(
        f"统一参数：imgsz={IMGSZ}, batch={BATCH}, conf={CONF}, "
        f"NMS IoU={NMS_IOU}, max_det={MAX_DET}, rect={RECT}, "
        f"augment={AUGMENT}, workers={WORKERS}",
        flush=True,
    )
    print("本脚本只调用 model.val()，不会启动训练。", flush=True)

    split_paths = {}
    for split in SPLITS:
        paths = resolve_split_path(data_cfg, split, DATA_YAML)
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(
                    f"{split} 数据路径不存在：{p}\n"
                    "请先检查 data.yaml 的 path/train/val 配置。"
                )
        image_count = count_images(paths)
        if image_count == 0:
            raise FileNotFoundError(
                f"{split} 路径中没有识别到图片：{paths}"
            )
        split_paths[split] = paths
        print(
            f"{split} 图片数量（预检查）：{image_count}；路径："
            + " ; ".join(str(p) for p in paths),
            flush=True,
        )
        expected = {"train": 1440, "val": 360}.get(split)
        if expected is not None and image_count != expected:
            print(
                f"警告：{split} 图片数为 {image_count}，"
                f"而当前项目记录的预期数量为 {expected}。"
                "请确认是否使用了正确的数据划分。",
                flush=True,
            )

    run_dir = OUTPUT_BASE / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    overall_rows = []
    per_class_rows = []

    for model_name, weight in WEIGHTS.items():
        for split in SPLITS:
            print("\n" + "=" * 78, flush=True)
            print(f"模型：{model_name}", flush=True)
            print(f"权重：{weight}", flush=True)
            print(f"数据划分：{split}", flush=True)
            print("=" * 78, flush=True)

            model = YOLO(str(weight))
            metrics = model.val(
                data=str(DATA_YAML),
                split=split,
                imgsz=IMGSZ,
                batch=BATCH,
                conf=CONF,
                iou=NMS_IOU,
                max_det=MAX_DET,
                device=DEVICE,
                workers=WORKERS,
                rect=RECT,
                augment=AUGMENT,
                plots=PLOTS,
                save_json=SAVE_JSON,
                verbose=True,
                project=str(run_dir),
                name=f"{model_name}_{split}_imgsz{IMGSZ}",
                exist_ok=False,
            )

            overall, classes = extract_metrics(
                metrics, model_name, split, names
            )
            overall_rows.append(overall)
            per_class_rows.extend(classes)

            print("\n本次官方汇总指标：", flush=True)
            print(f"  Precision: {overall['precision']}", flush=True)
            print(f"  Recall: {overall['recall']}", flush=True)
            print(f"  mAP50: {overall['mAP50']}", flush=True)
            print(f"  mAP50-95: {overall['mAP50_95']}", flush=True)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    overall_path = run_dir / "train_val_overall.csv"
    class_path = run_dir / "train_val_per_class.csv"
    write_csv(overall_path, overall_rows)
    write_csv(class_path, per_class_rows)

    print("\n" + "=" * 78, flush=True)
    print("train/val 官方验证对照完成。", flush=True)
    print(f"整体指标 CSV：{overall_path}", flush=True)
    print(f"六类指标 CSV：{class_path}", flush=True)
    print(f"Ultralytics 各次验证输出目录：{run_dir}", flush=True)
    print(
        "解释限制：训练集指标用于诊断；验证集已参与模型选择与参数比较，"
        "两者均不能替代独立测试或真实产线评估。",
        flush=True,
    )
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
