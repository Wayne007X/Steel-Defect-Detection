# -*- coding: utf-8 -*-
"""
NEU-DET 六类钢材表面缺陷：受控训练实验
实验名：neu_defect_mosaic000_cls075

目的：
在复现 YOLOv8n + Mosaic=0 基线的前提下，只将 cls 从 0.5 改为 0.75，
观察分类损失权重变化是否改善六类检测指标。

此脚本会：
1. 从原始 E:\\yolo_project\\yolov8n.pt 开始独立训练；
2. 保存到新的实验目录，不覆盖已有模型；
3. 训练后使用相同验证参数，对 clean、mosaic000、新模型进行标准验证；
4. 输出 overall_and_per_class_comparison.csv（整体和六类别指标）。

注意：
- 这是实验，不保证提升。
- 不会修改或覆盖现有权重。
- 若输出目录已存在，脚本会停止，避免意外覆盖。
"""

from pathlib import Path
import csv
import sys
import traceback

import numpy as np
from ultralytics import YOLO


# -------------------- 路径 --------------------
PROJECT_ROOT = Path(r"E:\yolo_project")
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
BASE_MODEL = PROJECT_ROOT / "yolov8n.pt"

CLEAN_BEST = RUNS_DIR / "neu_defect_clean" / "weights" / "best.pt"
MOSAIC000_BEST = RUNS_DIR / "neu_defect_mosaic000" / "weights" / "best.pt"

RUN_NAME = "neu_defect_mosaic000_cls075"
TRAIN_RUN_DIR = RUNS_DIR / RUN_NAME
VAL_PROJECT = RUNS_DIR / f"{RUN_NAME}_fair_val"
COMPARISON_CSV = RUNS_DIR / f"{RUN_NAME}_overall_and_per_class_comparison.csv"


# -------------------- 训练设置 --------------------
EPOCHS = 100
PATIENCE = 20
BATCH = 16
IMGSZ = 640
DEVICE = "0"
SEED = 42

BASE_CLS = 0.5
NEW_CLS = 0.75


def check_paths():
    required_files = [
        DATA_YAML,
        BASE_MODEL,
        CLEAN_BEST,
        MOSAIC000_BEST,
    ]
    missing = [str(p) for p in required_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "以下文件不存在，请检查路径：\n" + "\n".join(missing)
        )

    if TRAIN_RUN_DIR.exists():
        raise FileExistsError(
            f"训练输出目录已经存在，为避免覆盖已停止。\n"
            f"目录：{TRAIN_RUN_DIR}\n"
            f"如需重新进行实验，请修改脚本中的 RUN_NAME。"
        )
    if VAL_PROJECT.exists():
        raise FileExistsError(
            f"验证输出目录已经存在，为避免覆盖已停止。\n"
            f"目录：{VAL_PROJECT}\n"
            f"请修改脚本中的 RUN_NAME 后重新运行。"
        )
    if COMPARISON_CSV.exists():
        raise FileExistsError(
            f"对比 CSV 已存在，为避免覆盖已停止：{COMPARISON_CSV}\n"
            f"请修改脚本中的 RUN_NAME 后重新运行。"
        )


def extract_metrics(metrics, names, model_label):
    """从 Ultralytics DetMetrics 提取整体和分类别指标。"""
    rows = []
    box = metrics.box

    results_dict = getattr(metrics, "results_dict", {}) or {}
    overall_p = float(results_dict.get("metrics/precision(B)", getattr(box, "mp", 0.0)))
    overall_r = float(results_dict.get("metrics/recall(B)", getattr(box, "mr", 0.0)))
    overall_map50 = float(results_dict.get("metrics/mAP50(B)", getattr(box, "map50", 0.0)))
    overall_map = float(results_dict.get("metrics/mAP50-95(B)", getattr(box, "map", 0.0)))

    rows.append({
        "model": model_label,
        "class_id": "",
        "class_name": "all",
        "precision": overall_p,
        "recall": overall_r,
        "mAP50": overall_map50,
        "mAP50_95": overall_map,
    })

    class_ids = np.asarray(getattr(box, "ap_class_index", []), dtype=int)
    class_p = np.asarray(getattr(box, "p", []), dtype=float)
    class_r = np.asarray(getattr(box, "r", []), dtype=float)
    class_ap50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    class_ap = np.asarray(getattr(box, "ap", []), dtype=float)

    for pos, class_id in enumerate(class_ids):
        if isinstance(names, dict):
            class_name = str(names.get(int(class_id), class_id))
        else:
            class_name = str(names[int(class_id)])

        p = float(class_p[pos]) if pos < len(class_p) else 0.0
        r = float(class_r[pos]) if pos < len(class_r) else 0.0
        ap50 = float(class_ap50[pos]) if pos < len(class_ap50) else 0.0
        ap = float(class_ap[pos]) if pos < len(class_ap) else 0.0

        rows.append({
            "model": model_label,
            "class_id": int(class_id),
            "class_name": class_name,
            "precision": p,
            "recall": r,
            "mAP50": ap50,
            "mAP50_95": ap,
        })

    return rows


def run_fair_validation():
    models = [
        ("neu_defect_clean", CLEAN_BEST),
        ("neu_defect_mosaic000", MOSAIC000_BEST),
        (RUN_NAME, TRAIN_RUN_DIR / "weights" / "best.pt"),
    ]

    rows = []
    for label, weight_path in models:
        if not weight_path.is_file():
            raise FileNotFoundError(f"验证权重不存在：{weight_path}")

        print("\n" + "=" * 76)
        print(f"统一验证：{label}")
        print(f"权重：{weight_path}")
        print(f"imgsz={IMGSZ}, conf=0.001, iou=0.70")
        print("=" * 76)

        model = YOLO(str(weight_path))
        metrics = model.val(
            data=str(DATA_YAML),
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=0,
            conf=0.001,
            iou=0.70,
            max_det=300,
            plots=True,
            save_json=False,
            split="val",
            project=str(VAL_PROJECT),
            name=label,
            exist_ok=False,
            verbose=True,
        )

        rows.extend(extract_metrics(metrics, model.names, label))
        del model

    with COMPARISON_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "model", "class_id", "class_name",
            "precision", "recall", "mAP50", "mAP50_95",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("precision", "recall", "mAP50", "mAP50_95"):
                formatted[key] = round(float(formatted[key]), 6)
            writer.writerow(formatted)

    print("\n统一验证完成。指标对比 CSV：")
    print(COMPARISON_CSV)
    print("\n验证输出目录：")
    print(VAL_PROJECT)


def main():
    check_paths()

    print("=" * 76)
    print("NEU-DET 受控训练实验：mosaic=0，cls=0.75")
    print(f"数据配置：{DATA_YAML}")
    print(f"初始模型：{BASE_MODEL}")
    print(f"输出目录：{TRAIN_RUN_DIR}")
    print(f"参数：epochs={EPOCHS}, patience={PATIENCE}, imgsz={IMGSZ}, batch={BATCH}")
    print(f"基线 cls={BASE_CLS} -> 本次 cls={NEW_CLS}")
    print("其他主要训练参数沿用 mosaic000 基线设置。")
    print("=" * 76)

    model = YOLO(str(BASE_MODEL))
    model.train(
        task="detect",
        mode="train",
        model=str(BASE_MODEL),
        data=str(DATA_YAML),
        epochs=EPOCHS,
        time=None,
        patience=PATIENCE,
        batch=BATCH,
        imgsz=IMGSZ,
        save=True,
        save_period=-1,
        cache=False,
        device=DEVICE,
        workers=0,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=False,
        pretrained=True,
        cls_remap=True,
        optimizer="auto",
        verbose=True,
        seed=SEED,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        multi_scale=0.0,
        compile=False,
        val=True,
        split="val",
        save_json=False,
        conf=None,
        iou=0.7,
        max_det=300,
        plots=True,
        augment=False,
        agnostic_nms=False,
        classes=None,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=NEW_CLS,  # 本次唯一有意改变的主要训练参数
        cls_pw=0.0,
        dfl=1.5,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.4,
    )
    del model

    best_path = TRAIN_RUN_DIR / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(
            "训练调用结束，但没有找到 best.pt：\n" + str(best_path)
        )

    print("\n训练结束，最佳权重：")
    print(best_path)
    print("\n现在用同一份 data.yaml、imgsz=640 和统一验证参数比较三个模型。")
    run_fair_validation()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n脚本运行失败：")
        print(exc)
        traceback.print_exc()
        sys.exit(1)
