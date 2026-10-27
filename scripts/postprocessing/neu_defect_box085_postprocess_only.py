# -*- coding: utf-8 -*-
"""
NEU-DET 最后一项预先约定的受控训练实验：box=8.5

只有在阶段 1/2/3 结果显示定位相关错误值得研究后才运行本脚本。
本脚本只进行这一项训练实验，不会自动追加其他训练。

固定训练起点：E:\yolo_project\yolov8n.pt
唯一计划改变的训练超参数：box=8.5（基线为 7.5）
其余主要训练配置与 neu_defect_mosaic000 路线保持一致。

训练完成后自动：
1) 对 clean、mosaic000、box085 做相同条件的标准验证；
2) 在 NMS IoU=0.30 下，计算 conf=0.20/0.25 的固定工作点指标；
3) 在匹配 IoU=0.50/0.75 下分别统计；
4) 用相同规则统计验证集 FP/FN 错误类型；
5) 按预先规定的门槛打印本次实验是否达到目标。

环境：Windows、Ultralytics 8.4.x、PyTorch、单 GPU。
"""

from __future__ import annotations

import csv
import gc
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# ============================ 路径 ============================
PROJECT_ROOT = Path(r"E:\yolo_project")
RUNS_ROOT = PROJECT_ROOT / "runs"
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
PRETRAINED_WEIGHT = PROJECT_ROOT / "yolov8n.pt"

CLEAN_WEIGHT = RUNS_ROOT / "neu_defect_clean" / "weights" / "best.pt"
MOSAIC000_WEIGHT = RUNS_ROOT / "neu_defect_mosaic000" / "weights" / "best.pt"

TRAIN_NAME = "neu_defect_box085_controlled"
VAL_NAMES = {
    "neu_defect_clean": "neu_defect_box085_clean_fair_val",
    "neu_defect_mosaic000": "neu_defect_box085_mosaic000_fair_val",
    "neu_defect_box085": "neu_defect_box085_candidate_fair_val",
}

VAL_IMAGE_ROOT = Path(r"D:\yolov8\NEU-DET\validation\images")
VAL_LABEL_ROOT = Path(r"D:\yolov8\NEU-DET\validation\labels")

STANDARD_CSV = RUNS_ROOT / "neu_defect_box085_standard_validation_metrics.csv"
FIXED_OVERALL_CSV = RUNS_ROOT / "neu_defect_box085_fixed_threshold_overall.csv"
FIXED_CLASS_CSV = RUNS_ROOT / "neu_defect_box085_fixed_threshold_per_class.csv"
ERROR_TYPES_CSV = RUNS_ROOT / "neu_defect_box085_error_type_counts.csv"
DECISION_TXT = RUNS_ROOT / "neu_defect_box085_decision_report.txt"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ============================ 固定评估设置 ============================
IMGSZ = 640
VAL_BATCH = 2
PRED_BATCH = 2
DEVICE = 0
WORKERS = 0
VAL_FLOOR_CONF = 0.001
NMS_IOU_FOR_FIXED_METRICS = 0.30
GLOBAL_THRESHOLDS = (0.20, 0.25)
MATCH_IOUS = (0.50, 0.75)
MAX_DET = 300

# 仅用于补充分析的 per-class 阈值候选；来自本次已完成的验证集扫描，
# 因此会标记为“验证集内选定，可能偏乐观”，不作为实验成功与否的判据。
CLASS_THRESHOLDS_CANDIDATE = {
    0: 0.15,  # crazing
    1: 0.40,  # inclusion
    2: 0.40,  # patches
    3: 0.25,  # pitted_surface
    4: 0.25,  # rolled-in_scale
    5: 0.30,  # scratches
}


def log(message: str) -> None:
    print(message, flush=True)


def check_inputs_and_outputs() -> None:
    required = [
        DATA_YAML,
        PRETRAINED_WEIGHT,
        CLEAN_WEIGHT,
        MOSAIC000_WEIGHT,
        VAL_IMAGE_ROOT,
        VAL_LABEL_ROOT,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("以下必要路径不存在：\n" + "\n".join(missing))

    planned = [
        RUNS_ROOT / TRAIN_NAME,
        *(RUNS_ROOT / name for name in VAL_NAMES.values()),
        STANDARD_CSV,
        FIXED_OVERALL_CSV,
        FIXED_CLASS_CSV,
        ERROR_TYPES_CSV,
        DECISION_TXT,
    ]
    existing = [str(p) for p in planned if p.exists()]
    if existing:
        raise FileExistsError(
            "为防止覆盖已有结果，脚本停止。以下路径已存在：\n"
            + "\n".join(existing)
            + "\n如确需重跑，请修改脚本顶部的 TRAIN_NAME、VAL_NAMES 和输出 CSV 文件名。"
        )


def image_key(path: Path) -> str:
    """
    使用相对 validation/images 的路径作为唯一键。
    NEU-DET 图片按类别放在子目录中，例如 crazing/crazing_241.jpg。
    """
    try:
        return path.resolve().relative_to(VAL_IMAGE_ROOT.resolve()).as_posix().lower()
    except ValueError:
        # Ultralytics 返回的路径通常是绝对路径；若返回相对路径则保留其路径部分。
        return path.as_posix().lower()


def get_validation_images() -> list[Path]:
    # NEU-DET 的图片按类别分目录存放，必须递归查找。
    images = sorted(
        [p for p in VAL_IMAGE_ROOT.rglob("*")
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.relative_to(VAL_IMAGE_ROOT).as_posix().lower(),
    )
    if not images:
        raise FileNotFoundError(f"验证图片目录及其子目录中没有图片：{VAL_IMAGE_ROOT}")
    return images


def box_iou(a: dict, b: dict) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truths(images: list[Path]) -> dict[str, list[dict]]:
    gt_by_image: dict[str, list[dict]] = {}
    for image_path in images:
        key = image_key(image_path)
        # 标签目录与图片目录保持相同的类别子目录结构。
        relative_image = image_path.relative_to(VAL_IMAGE_ROOT)
        label_path = (VAL_LABEL_ROOT / relative_image).with_suffix(".txt")
        gts: list[dict] = []
        if label_path.exists():
            with Image.open(image_path) as im:
                width, height = im.size
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                if cid < 0 or cid >= len(CLASS_NAMES):
                    continue
                xc, yc, bw, bh = map(float, parts[1:5])
                gts.append({
                    "class_id": cid,
                    "class_name": CLASS_NAMES[cid],
                    "x1": (xc - bw / 2.0) * width,
                    "y1": (yc - bh / 2.0) * height,
                    "x2": (xc + bw / 2.0) * width,
                    "y2": (yc + bh / 2.0) * height,
                })
        gt_by_image[key] = gts
    return gt_by_image


def predict_validation(weight: Path, images: list[Path]) -> dict[str, list[dict]]:
    """对整个验证目录流式推理，使用低置信度下限缓存候选框，之后离线筛阈值。"""
    model = YOLO(str(weight))
    preds_by_image: dict[str, list[dict]] = {}
    results = model.predict(
        # Pass the recursively discovered image file list; Ultralytics directory loading is not recursive here.
        source=[str(p) for p in images],
        stream=True,
        imgsz=IMGSZ,
        batch=PRED_BATCH,
        conf=VAL_FLOOR_CONF,
        iou=NMS_IOU_FOR_FIXED_METRICS,
        max_det=MAX_DET,
        device=DEVICE,
        workers=WORKERS,
        save=False,
        verbose=False,
        augment=False,
    )
    expected = {image_key(p) for p in images}
    for idx, result in enumerate(results, 1):
        result_path = Path(result.path)
        if not result_path.is_absolute():
            result_path = (VAL_IMAGE_ROOT / result_path).resolve()
        key = image_key(result_path)
        if key not in expected:
            continue
        boxes = []
        if result.boxes is not None and len(result.boxes):
            coords = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cids = result.boxes.cls.cpu().numpy().astype(int)
            for xyxy, conf, cid in zip(coords, confs, cids):
                if 0 <= int(cid) < len(CLASS_NAMES):
                    boxes.append({
                        "class_id": int(cid),
                        "class_name": CLASS_NAMES[int(cid)],
                        "confidence": float(conf),
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                    })
        preds_by_image[key] = boxes
        if idx % 120 == 0 or idx == len(images):
            log(f"  推理进度：{idx}/{len(images)} 张")
    missing = sorted(expected - set(preds_by_image))
    if missing:
        raise RuntimeError(
            f"{len(missing)} 张验证图片未得到预测，示例：{missing[0]}"
        )
    del results, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds_by_image


def selected_predictions(preds: list[dict], threshold: float | dict[int, float]) -> list[dict]:
    out = []
    for pred in preds:
        min_conf = threshold.get(pred["class_id"], VAL_FLOOR_CONF) if isinstance(threshold, dict) else threshold
        if pred["confidence"] >= min_conf:
            out.append(pred)
    return sorted(out, key=lambda p: p["confidence"], reverse=True)


def evaluate_fixed(
    preds_by_image: dict[str, list[dict]],
    gt_by_image: dict[str, list[dict]],
    threshold: float | dict[int, float],
    match_iou: float,
) -> tuple[list[dict], dict]:
    counts = {
        cid: {"GT": 0, "TP": 0, "FP": 0, "FN": 0}
        for cid in range(len(CLASS_NAMES))
    }
    for key, gts in gt_by_image.items():
        for gt in gts:
            counts[gt["class_id"]]["GT"] += 1

        matched_gt: set[int] = set()
        preds = selected_predictions(preds_by_image.get(key, []), threshold)
        for pred in preds:
            candidates = [
                (gi, box_iou(pred, gt))
                for gi, gt in enumerate(gts)
                if gt["class_id"] == pred["class_id"] and gi not in matched_gt
            ]
            best = max(candidates, key=lambda x: x[1], default=(None, 0.0))
            if best[0] is not None and best[1] >= match_iou:
                matched_gt.add(best[0])
                counts[pred["class_id"]]["TP"] += 1
            else:
                counts[pred["class_id"]]["FP"] += 1

        for gi, gt in enumerate(gts):
            if gi not in matched_gt:
                counts[gt["class_id"]]["FN"] += 1

    class_rows = []
    f1s = []
    for cid, cname in enumerate(CLASS_NAMES):
        c = counts[cid]
        p = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else 0.0
        r = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        class_rows.append({
            "class_id": cid, "class_name": cname, **c,
            "precision": round(p, 6), "recall": round(r, 6), "F1": round(f1, 6),
        })

    tp = sum(c["TP"] for c in counts.values())
    fp = sum(c["FP"] for c in counts.values())
    fn = sum(c["FN"] for c in counts.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    overall = {
        "GT": sum(c["GT"] for c in counts.values()),
        "TP": tp, "FP": fp, "FN": fn,
        "micro_precision": round(p, 6),
        "micro_recall": round(r, 6),
        "micro_F1": round(f1, 6),
        "macro_F1": round(float(np.mean(f1s)), 6),
    }
    return class_rows, overall


def evaluate_error_types(
    preds_by_image: dict[str, list[dict]],
    gt_by_image: dict[str, list[dict]],
    threshold: float = 0.25,
    match_iou: float = 0.50,
) -> list[dict]:
    """
    对 clean、mosaic000、box085 使用同一规则计算错误类型。
    FP 分类优先级：重复/竞争 -> 同类定位相关 -> 类别混淆候选 -> 背景。
    FN 分类优先级：低置信度同类候选 -> 同类定位相关 -> 类别混淆候选 -> 无候选。
    这些是程序诊断类别，不是人工逐框确认的根因。
    """
    counts = {
        cid: {
            "GT": 0, "TP": 0, "FP": 0, "FN": 0,
            "FP_background": 0, "FP_localization_related": 0,
            "FP_duplicate_or_competing": 0, "FP_class_confusion_candidate": 0,
            "FN_low_confidence_candidate": 0, "FN_localization_related": 0,
            "FN_class_confusion_candidate": 0, "FN_no_candidate": 0,
        }
        for cid in range(len(CLASS_NAMES))
    }

    for key, gts in gt_by_image.items():
        for gt in gts:
            counts[gt["class_id"]]["GT"] += 1
        matched_gt: set[int] = set()
        all_preds = preds_by_image.get(key, [])
        chosen = selected_predictions(all_preds, threshold)

        for pred in chosen:
            cid = pred["class_id"]
            same_all = [(gi, box_iou(pred, gt)) for gi, gt in enumerate(gts) if gt["class_id"] == cid]
            unmatched_same = [(gi, iou) for gi, iou in same_all if gi not in matched_gt]
            best_unmatched = max(unmatched_same, key=lambda x: x[1], default=(None, 0.0))
            if best_unmatched[0] is not None and best_unmatched[1] >= match_iou:
                matched_gt.add(best_unmatched[0])
                counts[cid]["TP"] += 1
                continue

            counts[cid]["FP"] += 1
            any_same_overlap = max((iou for _, iou in same_all), default=0.0)
            any_same_matched_overlap = max(
                (iou for gi, iou in same_all if gi in matched_gt), default=0.0
            )
            other_overlap = max(
                (box_iou(pred, gt) for gt in gts if gt["class_id"] != cid),
                default=0.0,
            )

            if any_same_matched_overlap >= match_iou:
                counts[cid]["FP_duplicate_or_competing"] += 1
            elif any_same_overlap > 0.0:
                counts[cid]["FP_localization_related"] += 1
            elif other_overlap >= 0.10:
                counts[cid]["FP_class_confusion_candidate"] += 1
            else:
                counts[cid]["FP_background"] += 1

        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            cid = gt["class_id"]
            counts[cid]["FN"] += 1
            same_preds = [
                (pred, box_iou(pred, gt))
                for pred in all_preds if pred["class_id"] == cid
            ]
            low_candidate = any(iou > 0 and pred["confidence"] < threshold for pred, iou in same_preds)
            selected_overlap = any(iou > 0 and pred["confidence"] >= threshold for pred, iou in same_preds)
            other_class_candidate = any(
                box_iou(pred, gt) >= 0.10
                for pred in all_preds if pred["class_id"] != cid
            )
            if low_candidate:
                counts[cid]["FN_low_confidence_candidate"] += 1
            elif selected_overlap:
                counts[cid]["FN_localization_related"] += 1
            elif other_class_candidate:
                counts[cid]["FN_class_confusion_candidate"] += 1
            else:
                counts[cid]["FN_no_candidate"] += 1

    rows = []
    for cid, cname in enumerate(CLASS_NAMES):
        rows.append({"class_id": cid, "class_name": cname, **counts[cid]})
    return rows


def metric_rows(metrics, model_label: str, weight_path: Path) -> list[dict]:
    box = metrics.box
    class_ids = [int(x) for x in np.asarray(getattr(box, "ap_class_index", [])).reshape(-1)]
    p_values = np.asarray(getattr(box, "p", [])).reshape(-1)
    r_values = np.asarray(getattr(box, "r", [])).reshape(-1)
    ap50_values = np.asarray(getattr(box, "ap50", [])).reshape(-1)
    ap_values = np.asarray(getattr(box, "ap", [])).reshape(-1)

    rows = [{
        "model": model_label, "class_id": "all", "class_name": "all",
        "weight_path": str(weight_path), "imgsz": IMGSZ,
        "val_conf_floor": VAL_FLOOR_CONF, "val_nms_iou": 0.70,
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
        "precision_recall_note": "Ultralytics PR-curve operating point; not fixed-threshold P/R",
        "mAP50": round(float(box.map50), 6),
        "mAP50_95": round(float(box.map), 6),
    }]
    for cid, cname in enumerate(CLASS_NAMES):
        if cid in class_ids:
            idx = class_ids.index(cid)
            p = float(p_values[idx]) if idx < len(p_values) else 0.0
            r = float(r_values[idx]) if idx < len(r_values) else 0.0
            ap50 = float(ap50_values[idx]) if idx < len(ap50_values) else 0.0
            ap = float(ap_values[idx]) if idx < len(ap_values) else 0.0
        else:
            p = r = ap50 = ap = 0.0
        rows.append({
            "model": model_label, "class_id": cid, "class_name": cname,
            "weight_path": str(weight_path), "imgsz": IMGSZ,
            "val_conf_floor": VAL_FLOOR_CONF, "val_nms_iou": 0.70,
            "precision": round(p, 6), "recall": round(r, 6),
            "precision_recall_note": "Ultralytics per-class PR-curve operating point; not fixed-threshold P/R",
            "mAP50": round(ap50, 6), "mAP50_95": round(ap, 6),
        })
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"没有可写入的数据：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_candidate() -> Path:
    log("\n" + "=" * 76)
    log("开始唯一预先约定的训练实验：box=8.5")
    log("基线训练路线：YOLOv8n、640、mosaic=0、cls=0.5、seed=42")
    log("除 box=8.5 外，不新增其他训练改动。")
    log("=" * 76)

    model = YOLO(str(PRETRAINED_WEIGHT))
    model.train(
        task="detect",
        data=str(DATA_YAML),
        epochs=100,
        patience=20,
        batch=16,
        imgsz=640,
        save=True,
        save_period=-1,
        cache=False,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_ROOT),
        name=TRAIN_NAME,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
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
        iou=0.70,
        max_det=MAX_DET,
        plots=True,
        agnostic_nms=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=8.5,
        cls=0.5,
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_weight = RUNS_ROOT / TRAIN_NAME / "weights" / "best.pt"
    if not best_weight.is_file():
        raise FileNotFoundError(f"训练结束后未找到 best.pt：{best_weight}")
    return best_weight


def standard_validate(label: str, weight: Path) -> list[dict]:
    log("\n" + "=" * 76)
    log(f"统一标准验证：{label}")
    log(f"权重：{weight}")
    model = YOLO(str(weight))
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMGSZ,
        batch=VAL_BATCH,
        device=DEVICE,
        workers=WORKERS,
        conf=VAL_FLOOR_CONF,
        iou=0.70,
        max_det=MAX_DET,
        plots=True,
        save_json=False,
        project=str(RUNS_ROOT),
        name=VAL_NAMES[label],
        exist_ok=False,
        verbose=True,
    )
    rows = metric_rows(metrics, label, weight)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def write_decision_report(standard_rows: list[dict], fixed_rows: list[dict], error_rows: list[dict]) -> None:
    by_model_all = {
        row["model"]: row for row in standard_rows if str(row["class_id"]) == "all"
    }
    base = by_model_all["neu_defect_mosaic000"]
    candidate = by_model_all["neu_defect_box085"]

    def fixed_row(model: str, threshold: float, match_iou: float = 0.5):
        for row in fixed_rows:
            if (
                row["model"] == model and row["threshold_strategy"] == "global"
                and abs(float(row["confidence_threshold"]) - threshold) < 1e-9
                and abs(float(row["match_iou"]) - match_iou) < 1e-9
            ):
                return row
        return None

    base_fixed = fixed_row("neu_defect_mosaic000", 0.25, 0.5)
    cand_fixed = fixed_row("neu_defect_box085", 0.25, 0.5)

    base_cls = {
        row["class_name"]: row for row in standard_rows
        if row["model"] == "neu_defect_mosaic000" and str(row["class_id"]) != "all"
    }
    cand_cls = {
        row["class_name"]: row for row in standard_rows
        if row["model"] == "neu_defect_box085" and str(row["class_id"]) != "all"
    }
    ap50_deltas = {
        cname: float(cand_cls[cname]["mAP50"]) - float(base_cls[cname]["mAP50"])
        for cname in CLASS_NAMES
    }
    max_class_ap50_drop = min(ap50_deltas.values())

    base_loc = next(
        (r["FP_localization_related"] for r in error_rows
         if r["model"] == "neu_defect_mosaic000" and r["class_name"] == "rolled-in_scale"),
        None,
    )
    cand_loc = next(
        (r["FP_localization_related"] for r in error_rows
         if r["model"] == "neu_defect_box085" and r["class_name"] == "rolled-in_scale"),
        None,
    )

    delta_map95 = float(candidate["mAP50_95"]) - float(base["mAP50_95"])
    delta_map50 = float(candidate["mAP50"]) - float(base["mAP50"])
    delta_f1 = (
        float(cand_fixed["micro_F1"]) - float(base_fixed["micro_F1"])
        if base_fixed and cand_fixed else float("nan")
    )

    gates = {
        "mAP50_95 improvement >= 0.010": delta_map95 >= 0.010,
        "mAP50 drop no worse than -0.005": delta_map50 >= -0.005,
        "fixed-threshold micro-F1 drop no worse than -0.010": delta_f1 >= -0.010,
        "no class mAP50 drop exceeds 0.030": max_class_ap50_drop >= -0.030,
        "rolled-in_scale localization-related FP decreased": (
            base_loc is not None and cand_loc is not None and cand_loc < base_loc
        ),
    }
    passed = all(gates.values())

    lines = [
        "NEU-DET box=8.5 controlled experiment decision report",
        "",
        f"clean mAP50={by_model_all['neu_defect_clean']['mAP50']:.6f}, "
        f"mAP50_95={by_model_all['neu_defect_clean']['mAP50_95']:.6f}",
        f"mosaic000 mAP50={base['mAP50']:.6f}, mAP50_95={base['mAP50_95']:.6f}",
        f"box085 mAP50={candidate['mAP50']:.6f}, mAP50_95={candidate['mAP50_95']:.6f}",
        f"delta mAP50={delta_map50:+.6f}",
        f"delta mAP50_95={delta_map95:+.6f}",
        f"mosaic000 fixed F1 (conf=.25, NMS IoU=.30, match IoU=.50)="
        f"{base_fixed['micro_F1'] if base_fixed else float('nan')}",
        f"box085 fixed F1 (conf=.25, NMS IoU=.30, match IoU=.50)="
        f"{cand_fixed['micro_F1'] if cand_fixed else float('nan')}",
        f"delta fixed F1={delta_f1:+.6f}",
        f"rolled-in_scale localization-related FP: mosaic000={base_loc}, box085={cand_loc}",
        "",
        "Per-class mAP50 deltas (box085 - mosaic000):",
    ]
    lines.extend([f"  {k}: {v:+.6f}" for k, v in ap50_deltas.items()])
    lines.append("")
    lines.append("Predefined acceptance gates:")
    lines.extend([f"  {'PASS' if ok else 'FAIL'}: {name}" for name, ok in gates.items()])
    lines.append("")
    lines.append(
        "OVERALL: " + ("MEETS ALL PREDEFINED GATES" if passed else "DOES NOT MEET ALL PREDEFINED GATES")
    )
    lines.append(
        "Do not launch another training experiment if this run does not meet all gates. "
        "Thresholds selected on the validation set are provisional, not independent test results."
    )
    DECISION_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


def main() -> None:
    """
    仅续跑训练后的诊断推理与 CSV 输出。
    不训练、不重新验证、不覆盖 best.pt。
    """
    images = get_validation_images()
    gt_by_image = load_ground_truths(images)
    log(f"验证图片：{len(images)} 张")
    log(f"验证集真实目标：{sum(len(v) for v in gt_by_image.values())} 个")

    candidate_weight = RUNS_ROOT / TRAIN_NAME / "weights" / "best.pt"
    models = [
        ("neu_defect_clean", CLEAN_WEIGHT),
        ("neu_defect_mosaic000", MOSAIC000_WEIGHT),
        ("neu_defect_box085", candidate_weight),
    ]
    for label, weight in models:
        if not weight.is_file():
            raise FileNotFoundError(f"缺少模型权重：{label}: {weight}")

    # Reuse the standard-validation CSV created before the previous run failed.
    if not STANDARD_CSV.is_file():
        raise FileNotFoundError(
            f"未找到已完成的标准验证指标文件：{STANDARD_CSV}\n"
            "不要重新训练；请先确认该 CSV 是否存在。"
        )
    with STANDARD_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        standard_rows = list(csv.DictReader(f))
    required_models = {"neu_defect_clean", "neu_defect_mosaic000", "neu_defect_box085"}
    found_models = {row.get("model", "") for row in standard_rows}
    if not required_models.issubset(found_models):
        raise RuntimeError(
            "标准验证 CSV 未包含 clean、mosaic000、box085 三个模型，"
            "为避免混合不完整结果，停止后处理。"
        )

    fixed_overall_rows = []
    fixed_class_rows = []
    error_rows = []

    for label, weight in models:
        log("\n" + "=" * 76)
        log(f"仅执行诊断推理：{label}")
        log(f"权重：{weight}")
        preds = predict_validation(weight, images)

        for threshold in GLOBAL_THRESHOLDS:
            for match_iou in MATCH_IOUS:
                class_rows, overall = evaluate_fixed(
                    preds, gt_by_image, threshold=threshold, match_iou=match_iou
                )
                fixed_overall_rows.append({
                    "model": label,
                    "nms_iou": NMS_IOU_FOR_FIXED_METRICS,
                    "threshold_strategy": "global",
                    "confidence_threshold": threshold,
                    "match_iou": match_iou,
                    **overall,
                })
                for row in class_rows:
                    fixed_class_rows.append({
                        "model": label,
                        "nms_iou": NMS_IOU_FOR_FIXED_METRICS,
                        "threshold_strategy": "global",
                        "confidence_threshold": threshold,
                        "match_iou": match_iou,
                        **row,
                    })

        for match_iou in MATCH_IOUS:
            class_rows, overall = evaluate_fixed(
                preds, gt_by_image,
                threshold=CLASS_THRESHOLDS_CANDIDATE,
                match_iou=match_iou,
            )
            fixed_overall_rows.append({
                "model": label,
                "nms_iou": NMS_IOU_FOR_FIXED_METRICS,
                "threshold_strategy": "per_class_validation_candidate",
                "confidence_threshold": "",
                "per_class_thresholds": (
                    "crazing=0.15; inclusion=0.40; patches=0.40; "
                    "pitted_surface=0.25; rolled-in_scale=0.25; scratches=0.30"
                ),
                "match_iou": match_iou,
                **overall,
            })
            for row in class_rows:
                fixed_class_rows.append({
                    "model": label,
                    "nms_iou": NMS_IOU_FOR_FIXED_METRICS,
                    "threshold_strategy": "per_class_validation_candidate",
                    "confidence_threshold": CLASS_THRESHOLDS_CANDIDATE[row["class_id"]],
                    "per_class_thresholds": (
                        "crazing=0.15; inclusion=0.40; patches=0.40; "
                        "pitted_surface=0.25; rolled-in_scale=0.25; scratches=0.30"
                    ),
                    "match_iou": match_iou,
                    **row,
                })

        err = evaluate_error_types(
            preds, gt_by_image, threshold=0.25, match_iou=0.50
        )
        for row in err:
            error_rows.append({
                "model": label,
                "nms_iou": NMS_IOU_FOR_FIXED_METRICS,
                "confidence_threshold": 0.25,
                "match_iou": 0.50,
                **row,
            })

        del preds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_csv(FIXED_OVERALL_CSV, fixed_overall_rows)
        save_csv(FIXED_CLASS_CSV, fixed_class_rows)
        save_csv(ERROR_TYPES_CSV, error_rows)

    write_decision_report(standard_rows, fixed_overall_rows, error_rows)

    log("\n" + "=" * 76)
    log("训练后的诊断推理与对照已完成。没有启动任何训练。")
    log(f"已有训练权重：{candidate_weight}")
    log(f"标准验证指标（复用已有文件）：{STANDARD_CSV}")
    log(f"固定阈值总体指标：{FIXED_OVERALL_CSV}")
    log(f"固定阈值六类指标：{FIXED_CLASS_CSV}")
    log(f"错误类型对照：{ERROR_TYPES_CSV}")
    log(f"预设门槛判定：{DECISION_TXT}")
    log("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n脚本运行失败：")
        print(exc)
        traceback.print_exc()
        sys.exit(1)
