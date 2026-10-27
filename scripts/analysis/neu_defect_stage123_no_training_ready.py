# -*- coding: utf-8 -*-
"""
NEU-DET 六类 YOLOv8：阶段 1/2/3 自动化流程（绝不训练）

阶段 1：对现有候选权重做统一标准验证（640、相同 data.yaml、相同验证参数）。
阶段 2：对 clean 与 mosaic000 做完整的 NMS IoU × confidence 网格比较；
        每个 NMS IoU 只推理一次（conf=0.001），再离线扫描 confidence。
阶段 3：读取已有 train/val 错误分析 CSV，导出错误类型汇总，不重复训练集推理。

适用环境：Windows、Ultralytics 8.4.x、单 GPU。默认 batch=2，兼顾 6GB 显存。
本脚本只调用 model.val() / model.predict()，不调用 model.train()。
"""
from __future__ import annotations

import csv
import gc
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from ultralytics import YOLO

# ============================ 固定路径 ============================
PROJECT = Path(r"E:\yolo_project")
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
VAL_IMAGE_ROOT = Path(r"D:\yolov8\NEU-DET\validation\images")
VAL_LABEL_ROOT = Path(r"D:\yolov8\NEU-DET\validation\labels")

# 输出目录已存在时脚本会停止，防止覆盖旧结果。重跑时只需修改末尾标记。
OUTPUT_ROOT = PROJECT / "runs" / "neu_defect_tuning_stage123_no_training_20261026"

ERROR_ANALYSIS_ROOT = PROJECT / "runs" / "neu_defect_train_val_error_comparison_v2"

CLASS_NAMES = [
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled-in_scale", "scratches",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ============================ 评估设置 ============================
IMGSZ = 640
VAL_BATCH = 2
PRED_BATCH = 2
DEVICE = "0"
WORKERS = 0
PRED_MIN_CONF = 0.001
MAX_DET = 300

# 阶段 2：计划内固定网格，不在结果出来后追加新参数。
CONF_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
NMS_IOU_THRESHOLDS = [0.30, 0.45, 0.50, 0.60, 0.70]
MATCH_IOU_THRESHOLDS = [0.50, 0.75]

# 需要统一验证的已训练候选。不存在的可选权重会记录并跳过；clean/mosaic000 为必需。
MODEL_CANDIDATES = [
    ("neu_defect_clean", [
        PROJECT / "runs" / "neu_defect_clean" / "weights" / "best.pt",
    ]),
    ("neu_defect_mosaic050", [
        PROJECT / "runs" / "neu_defect_mosaic050" / "weights" / "best.pt",
    ]),
    ("neu_defect_mosaic000", [
        PROJECT / "runs" / "neu_defect_mosaic000" / "weights" / "best.pt",
    ]),
    ("neu_defect_yolov8s_mosaic000", [
        PROJECT / "runs" / "neu_defect_yolov8s_mosaic000" / "weights" / "best.pt",
    ]),
    ("neu_defect_mosaic000_img800_ft", [
        PROJECT / "runs" / "neu_defect_mosaic000_img800_ft" / "weights" / "best.pt",
    ]),
    ("neu_defect_mosaic000_cls075_controlled", [
        PROJECT / "runs" / "neu_defect_mosaic000_cls075_controlled" / "weights" / "best.pt",
        PROJECT / "runs" / "neu_defect_mosaic000_cls075" / "weights" / "best.pt",
    ]),
    ("neu_defect_crazing_weighted150", [
        PROJECT / "runs" / "neu_defect_crazing_weighted150" / "weights" / "best.pt",
    ]),
    ("full_six_class_tuning_mosaic000", [
        PROJECT / "runs" / "full_six_class_tuning_mosaic000" / "weights" / "best.pt",
        PROJECT / "runs" / "detect" / "full_six_class_tuning_mosaic000" / "weights" / "best.pt",
    ]),
    ("full_six_class_tuning_mosaic050", [
        PROJECT / "runs" / "full_six_class_tuning_mosaic050" / "weights" / "best.pt",
        PROJECT / "runs" / "detect" / "full_six_class_tuning_mosaic050" / "weights" / "best.pt",
    ]),
]
STAGE2_MODELS = ["neu_defect_clean", "neu_defect_mosaic000"]


def log(msg: str) -> None:
    print(msg, flush=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_models() -> tuple[list[tuple[str, Path]], list[dict]]:
    found, missing = [], []
    for name, candidates in MODEL_CANDIDATES:
        existing = next((p for p in candidates if p.is_file()), None)
        if existing is None:
            missing.append({"model": name, "checked_paths": " ; ".join(str(p) for p in candidates)})
            continue
        found.append((name, existing))
    names = {name for name, _ in found}
    required = {"neu_defect_clean", "neu_defect_mosaic000"}
    if not required.issubset(names):
        absent = sorted(required - names)
        raise FileNotFoundError(
            "缺少阶段 1/2 必需权重：" + ", ".join(absent) +
            "。请先确认这些 best.pt 路径，不要开始训练。"
        )
    return found, missing


def get_class_metric_rows(metrics, model_name: str, weight_path: Path) -> list[dict]:
    box = metrics.box
    class_ids = [int(x) for x in np.asarray(getattr(box, "ap_class_index", [])).reshape(-1)]
    p_arr = np.asarray(getattr(box, "p", [])).reshape(-1)
    r_arr = np.asarray(getattr(box, "r", [])).reshape(-1)
    ap50_arr = np.asarray(getattr(box, "ap50", [])).reshape(-1)
    ap_arr = np.asarray(getattr(box, "ap", [])).reshape(-1)

    rows = [{
        "model": model_name, "class_id": "all", "class_name": "all",
        "weight_path": str(weight_path), "imgsz": IMGSZ,
        "val_conf": PRED_MIN_CONF, "nms_iou": 0.70, "max_det": MAX_DET,
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
        "precision_recall_note": "Ultralytics validation P/R at its PR-curve operating point; conf=0.001 is the low prediction floor, not a fixed operating threshold",
        "mAP50": round(float(box.map50), 6),
        "mAP50_95": round(float(box.map), 6),
    }]
    for cid, cname in enumerate(CLASS_NAMES):
        if cid in class_ids:
            idx = class_ids.index(cid)
            p = float(p_arr[idx]) if idx < len(p_arr) else 0.0
            r = float(r_arr[idx]) if idx < len(r_arr) else 0.0
            ap50 = float(ap50_arr[idx]) if idx < len(ap50_arr) else 0.0
            ap = float(ap_arr[idx]) if idx < len(ap_arr) else 0.0
        else:
            p = r = ap50 = ap = 0.0
        rows.append({
            "model": model_name, "class_id": cid, "class_name": cname,
            "weight_path": str(weight_path), "imgsz": IMGSZ,
            "val_conf": PRED_MIN_CONF, "nms_iou": 0.70, "max_det": MAX_DET,
            "precision": round(p, 6), "recall": round(r, 6),
            "precision_recall_note": "Ultralytics validation per-class P/R at its PR-curve operating point; not fixed-threshold P/R",
            "mAP50": round(ap50, 6), "mAP50_95": round(ap, 6),
        })
    return rows


def run_stage1(models: list[tuple[str, Path]]) -> list[dict]:
    log("\n" + "=" * 76)
    log("阶段 1：统一标准验证（只验证，不训练）")
    log(f"data.yaml={DATA_YAML}; imgsz={IMGSZ}; batch={VAL_BATCH}; conf={PRED_MIN_CONF}; NMS IoU=0.70; max_det={MAX_DET}")
    log("=" * 76)
    all_rows = []
    val_project = OUTPUT_ROOT / "standard_validation_runs"
    for name, weight in models:
        log(f"\n验证模型：{name}\n权重：{weight}")
        model = YOLO(str(weight))
        metrics = model.val(
            data=str(DATA_YAML), split="val", imgsz=IMGSZ,
            batch=VAL_BATCH, device=DEVICE, workers=WORKERS,
            conf=PRED_MIN_CONF, iou=0.70, max_det=MAX_DET,
            plots=True, save_json=False, verbose=True,
            project=str(val_project), name=f"val_{name}", exist_ok=False,
        )
        all_rows.extend(get_class_metric_rows(metrics, name, weight))
        del model, metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_csv(OUTPUT_ROOT / "standard_validation_metrics.csv", all_rows)
    write_csv(OUTPUT_ROOT / "standard_validation_metrics.csv", all_rows)
    log(f"阶段 1 完成：{OUTPUT_ROOT / 'standard_validation_metrics.csv'}")
    return all_rows


def image_key(path: Path) -> str:
    return str(path.resolve()).lower()


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_segment = next((i for i, part in enumerate(parts) if part.lower() == "images"), None)
    if image_segment is None:
        raise ValueError(f"图片路径中找不到 images 目录段：{image_path}")
    parts[image_segment] = "labels"
    return Path(*parts).with_suffix(".txt")


def collect_validation_ground_truths() -> tuple[list[Path], dict[str, list[dict]]]:
    if not VAL_IMAGE_ROOT.is_dir() or not VAL_LABEL_ROOT.is_dir():
        raise FileNotFoundError(f"验证图片或标签目录不存在：{VAL_IMAGE_ROOT} / {VAL_LABEL_ROOT}")
    images = sorted(
        (p for p in VAL_IMAGE_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    if not images:
        raise FileNotFoundError(f"没有找到验证图片：{VAL_IMAGE_ROOT}")
    gt_by_image: dict[str, list[dict]] = {}
    missing = []
    for image_path in images:
        label_path = label_path_for_image(image_path)
        if not label_path.is_file():
            missing.append(str(label_path))
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        text = label_path.read_text(encoding="utf-8-sig").strip()
        gts = []
        if text:
            for line_no, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"标签格式错误：{label_path} 第 {line_no} 行")
                cid = int(float(parts[0]))
                if cid < 0 or cid >= len(CLASS_NAMES):
                    raise ValueError(f"类别 ID 越界：{label_path} 第 {line_no} 行 class={cid}")
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = max(0.0, (xc - bw / 2) * width)
                y1 = max(0.0, (yc - bh / 2) * height)
                x2 = min(float(width), (xc + bw / 2) * width)
                y2 = min(float(height), (yc + bh / 2) * height)
                gts.append({"class_id": cid, "class_name": CLASS_NAMES[cid],
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        gt_by_image[image_key(image_path)] = gts
    if missing:
        raise FileNotFoundError(
            f"有 {len(missing)} 张验证图片找不到标签。脚本不会把缺标签当作背景图。示例：{missing[0]}"
        )
    log(f"验证图片：{len(images)} 张；真实目标：{sum(len(v) for v in gt_by_image.values())} 个")
    return images, gt_by_image


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


def predict_one_nms_setting(model: YOLO, model_name: str, images: list[Path], nms_iou: float) -> dict[str, list[dict]]:
    """按图片父目录流式推理，避免把数百张图片路径作为一个大批次传入造成 OOM。"""
    preds_by_image: dict[str, list[dict]] = {}
    expected_by_parent: dict[Path, set[str]] = defaultdict(set)
    for p in images:
        expected_by_parent[p.parent].add(image_key(p))
    processed = 0
    for parent in sorted(expected_by_parent, key=lambda p: str(p).lower()):
        expected = expected_by_parent[parent]
        results = model.predict(
            source=str(parent), stream=True, imgsz=IMGSZ,
            batch=PRED_BATCH, conf=PRED_MIN_CONF, iou=nms_iou,
            max_det=MAX_DET, device=DEVICE, workers=WORKERS,
            save=False, verbose=False, augment=False,
        )
        for result in results:
            key = image_key(Path(result.path))
            if key not in expected:
                continue
            boxes = []
            if result.boxes is not None and len(result.boxes):
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                ids = result.boxes.cls.cpu().numpy().astype(int)
                for coords, conf, cid in zip(xyxy, confs, ids):
                    if 0 <= int(cid) < len(CLASS_NAMES):
                        boxes.append({
                            "class_id": int(cid), "class_name": CLASS_NAMES[int(cid)],
                            "confidence": float(conf), "x1": float(coords[0]),
                            "y1": float(coords[1]), "x2": float(coords[2]), "y2": float(coords[3]),
                        })
            preds_by_image[key] = boxes
            processed += 1
            if processed % 120 == 0 or processed == len(images):
                log(f"    NMS IoU={nms_iou:.2f}: 推理 {processed}/{len(images)} 张")
        del results
    missing = [p for p in images if image_key(p) not in preds_by_image]
    if missing:
        raise RuntimeError(f"有 {len(missing)} 张图片未获得预测结果。示例：{missing[0]}")
    return preds_by_image


def count_metrics(preds_by_image: dict[str, list[dict]], gt_by_image: dict[str, list[dict]],
                  threshold: float | dict[int, float], match_iou: float) -> tuple[list[dict], dict]:
    counts = {cid: {"GT": 0, "TP": 0, "FP": 0, "FN": 0} for cid in range(len(CLASS_NAMES))}
    for key, gts in gt_by_image.items():
        for gt in gts:
            counts[gt["class_id"]]["GT"] += 1
        selected = []
        for pred in preds_by_image.get(key, []):
            if isinstance(threshold, dict):
                min_conf = threshold.get(pred["class_id"], PRED_MIN_CONF)
            else:
                min_conf = threshold
            if pred["confidence"] >= min_conf:
                selected.append(pred)
        selected.sort(key=lambda p: p["confidence"], reverse=True)
        matched_gt: set[int] = set()
        for pred in selected:
            best_idx, best_overlap = None, -1.0
            for gi, gt in enumerate(gts):
                if gi in matched_gt or gt["class_id"] != pred["class_id"]:
                    continue
                overlap = box_iou(pred, gt)
                if overlap >= match_iou and overlap > best_overlap:
                    best_idx, best_overlap = gi, overlap
            if best_idx is None:
                counts[pred["class_id"]]["FP"] += 1
            else:
                matched_gt.add(best_idx)
                counts[pred["class_id"]]["TP"] += 1
        for gi, gt in enumerate(gts):
            if gi not in matched_gt:
                counts[gt["class_id"]]["FN"] += 1

    class_rows = []
    f1s = []
    for cid, cname in enumerate(CLASS_NAMES):
        c = counts[cid]
        precision = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else 0.0
        recall = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        class_rows.append({
            "class_id": cid, "class_name": cname, **c,
            "precision": round(precision, 6), "recall": round(recall, 6), "F1": round(f1, 6),
        })
    total_tp = sum(c["TP"] for c in counts.values())
    total_fp = sum(c["FP"] for c in counts.values())
    total_fn = sum(c["FN"] for c in counts.values())
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    overall = {
        "GT": sum(c["GT"] for c in counts.values()), "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "micro_precision": round(micro_p, 6), "micro_recall": round(micro_r, 6),
        "micro_F1": round(micro_f1, 6), "macro_F1": round(float(np.mean(f1s)), 6),
    }
    return class_rows, overall


def append_raw_predictions(path: Path, model_name: str, nms_iou: float,
                           preds_by_image: dict[str, list[dict]], first_write: bool) -> None:
    fields = ["model", "nms_iou", "image", "prediction_index", "class_id", "class_name",
              "confidence", "x1", "y1", "x2", "y2"]
    mode = "w" if first_write else "a"
    with path.open(mode, newline="", encoding="utf-8-sig" if first_write else "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if first_write:
            writer.writeheader()
        for image, boxes in preds_by_image.items():
            for idx, box in enumerate(boxes):
                writer.writerow({"model": model_name, "nms_iou": nms_iou, "image": image,
                                 "prediction_index": idx, **box})


def run_stage2(models: list[tuple[str, Path]]) -> None:
    log("\n" + "=" * 76)
    log("阶段 2：clean 与 mosaic000 的完整推理参数校准（只推理，不训练）")
    log(f"conf={CONF_THRESHOLDS}; NMS IoU={NMS_IOU_THRESHOLDS}; match IoU={MATCH_IOU_THRESHOLDS}")
    log("每个模型/每个 NMS IoU 仅推理一次，之后离线扫描 confidence。")
    log("=" * 76)

    images, gt_by_image = collect_validation_ground_truths()
    by_name = dict(models)
    raw_path = OUTPUT_ROOT / "raw_prediction_cache.csv"
    if raw_path.exists():
        raise FileExistsError(f"预测缓存文件已存在，为避免覆盖：{raw_path}")
    first_raw_write = True
    overall_rows, class_rows_all = [], []
    best_global_rows, best_class_rows, combined_rows = [], [], []

    for model_name in STAGE2_MODELS:
        if model_name not in by_name:
            raise FileNotFoundError(f"阶段 2 缺少模型：{model_name}")
        weight = by_name[model_name]
        log(f"\n阶段 2 模型：{model_name} ({weight})")
        model = YOLO(str(weight))
        for nms_iou in NMS_IOU_THRESHOLDS:
            log(f"\n推理：{model_name}，NMS IoU={nms_iou:.2f}")
            preds_by_image = predict_one_nms_setting(model, model_name, images, nms_iou)
            append_raw_predictions(raw_path, model_name, nms_iou, preds_by_image, first_raw_write)
            first_raw_write = False

            for conf in CONF_THRESHOLDS:
                for match_iou in MATCH_IOU_THRESHOLDS:
                    class_metrics, overall = count_metrics(preds_by_image, gt_by_image, conf, match_iou)
                    overall_rows.append({"model": model_name, "nms_iou": nms_iou,
                                         "confidence_threshold": conf, "match_iou": match_iou, **overall})
                    for row in class_metrics:
                        class_rows_all.append({"model": model_name, "nms_iou": nms_iou,
                                               "confidence_threshold": conf, "match_iou": match_iou, **row})

            # 每个 model × NMS IoU 的最佳全局阈值：先看 macro F1，再看 micro F1；只按 match IoU=0.50 选择。
            candidate_overall = [r for r in overall_rows if r["model"] == model_name and
                                 abs(r["nms_iou"] - nms_iou) < 1e-9 and abs(r["match_iou"] - 0.50) < 1e-9]
            best_global = max(candidate_overall, key=lambda r: (r["macro_F1"], r["micro_F1"], r["confidence_threshold"]))
            best_global_rows.append(best_global.copy())

            # 每类阈值：在 match IoU=0.50 下最大化该类 F1；并列时偏向 precision 更高、阈值更高者。
            current_class_sweep = [r for r in class_rows_all if r["model"] == model_name and
                                    abs(r["nms_iou"] - nms_iou) < 1e-9 and abs(r["match_iou"] - 0.50) < 1e-9]
            selected_thresholds: dict[int, float] = {}
            for cid, cname in enumerate(CLASS_NAMES):
                rows_c = [r for r in current_class_sweep if int(r["class_id"]) == cid]
                best_c = max(rows_c, key=lambda r: (r["F1"], r["precision"], r["confidence_threshold"]))
                selected_thresholds[cid] = float(best_c["confidence_threshold"])
                best_class_rows.append({"model": model_name, "nms_iou": nms_iou,
                                        "selected_by": "per-class max F1 on validation split",
                                        **best_c})

            for match_iou in MATCH_IOU_THRESHOLDS:
                combined_class_metrics, combined_overall = count_metrics(
                    preds_by_image, gt_by_image, selected_thresholds, match_iou
                )
                combined_rows.append({"model": model_name, "nms_iou": nms_iou,
                                      "threshold_strategy": "per-class F1-selected on same validation split",
                                      "per_class_thresholds": "; ".join(
                                          f"{CLASS_NAMES[cid]}={selected_thresholds[cid]:.2f}" for cid in range(6)),
                                      "match_iou": match_iou, "class_id": "all", "class_name": "all",
                                      **combined_overall})
                for row in combined_class_metrics:
                    combined_rows.append({"model": model_name, "nms_iou": nms_iou,
                                          "threshold_strategy": "per-class F1-selected on same validation split",
                                          "per_class_thresholds": "; ".join(
                                              f"{CLASS_NAMES[cid]}={selected_thresholds[cid]:.2f}" for cid in range(6)),
                                          "match_iou": match_iou, **row})

            write_csv(OUTPUT_ROOT / "threshold_sweep_overall.csv", overall_rows)
            write_csv(OUTPUT_ROOT / "threshold_sweep_per_class.csv", class_rows_all)
            write_csv(OUTPUT_ROOT / "best_global_thresholds.csv", best_global_rows)
            write_csv(OUTPUT_ROOT / "best_per_class_thresholds.csv", best_class_rows)
            write_csv(OUTPUT_ROOT / "per_class_threshold_combined_metrics.csv", combined_rows)
            del preds_by_image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"阶段 2 完成：{OUTPUT_ROOT}")


def run_stage3_snapshot() -> None:
    log("\n" + "=" * 76)
    log("阶段 3：整理已有训练集/验证集错误诊断（不重新推理）")
    log("=" * 76)
    metrics_path = ERROR_ANALYSIS_ROOT / "metrics_summary.csv"
    gap_path = ERROR_ANALYSIS_ROOT / "train_val_gap.csv"
    gt_path = ERROR_ANALYSIS_ROOT / "gt_error_details.csv"
    pred_path = ERROR_ANALYSIS_ROOT / "prediction_details.csv"

    if metrics_path.is_file():
        df = pd.read_csv(metrics_path)
        wanted = [c for c in [
            "model", "split", "class_id", "class_name", "images", "GT", "TP", "FP", "FN",
            "precision", "recall", "F1", "FP_background", "FP_localization_related",
            "FP_duplicate_or_competing", "FP_class_confusion", "FN_low_confidence_candidate",
            "FN_localization_related", "FN_class_confusion", "FN_no_candidate", "FN_duplicate_or_competing",
        ] if c in df.columns]
        write_csv(OUTPUT_ROOT / "stage3_existing_metrics_summary.csv", df[wanted].to_dict("records"), wanted)
        log(f"已读取：{metrics_path}")
    else:
        log(f"提示：未找到现有错误汇总，跳过：{metrics_path}")

    if gap_path.is_file():
        gap = pd.read_csv(gap_path)
        write_csv(OUTPUT_ROOT / "stage3_existing_train_val_gap.csv", gap.to_dict("records"), list(gap.columns))
        log(f"已读取：{gap_path}")
    else:
        log(f"提示：未找到 train/val gap，跳过：{gap_path}")

    if gt_path.is_file():
        gt = pd.read_csv(gt_path)
        if {"model", "split", "class_name", "gt_error_type"}.issubset(gt.columns):
            summary = (gt[gt["gt_status"].astype(str).eq("FN")]
                       .groupby(["model", "split", "class_name", "gt_error_type"], dropna=False)
                       .size().reset_index(name="count"))
            write_csv(OUTPUT_ROOT / "stage3_gt_error_type_counts.csv", summary.to_dict("records"), list(summary.columns))
        log(f"已读取：{gt_path}")
    else:
        log(f"提示：未找到 GT 明细，跳过：{gt_path}")

    if pred_path.is_file():
        pred = pd.read_csv(pred_path)
        needed = {"model", "split", "class_name", "prediction_error_type", "status_at_conf_025"}
        if needed.issubset(pred.columns):
            fp = pred[pred["status_at_conf_025"].astype(str).eq("FP")]
            summary = (fp.groupby(["model", "split", "class_name", "prediction_error_type"], dropna=False)
                       .size().reset_index(name="count"))
            write_csv(OUTPUT_ROOT / "stage3_prediction_error_type_counts.csv", summary.to_dict("records"), list(summary.columns))
        log(f"已读取：{pred_path}")
    else:
        log(f"提示：未找到预测明细，跳过：{pred_path}")


def main() -> None:
    for path, label in [(DATA_YAML, "data.yaml"), (VAL_IMAGE_ROOT, "验证图片目录"), (VAL_LABEL_ROOT, "验证标签目录")]:
        if not path.exists():
            raise FileNotFoundError(f"找不到{label}：{path}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"输出目录已存在，为避免覆盖旧结果，脚本停止：{OUTPUT_ROOT}\n"
            "如果确实需要重跑，请修改脚本顶部 OUTPUT_ROOT 的目录名。"
        )
    models, missing = resolve_models()
    OUTPUT_ROOT.mkdir(parents=True)
    write_csv(OUTPUT_ROOT / "missing_optional_model_weights.csv", missing,
              ["model", "checked_paths"])
    write_csv(OUTPUT_ROOT / "resolved_model_weights.csv",
              [{"model": n, "weight_path": str(p)} for n, p in models],
              ["model", "weight_path"])
    log("=" * 76)
    log("NEU-DET 阶段 1/2/3：仅验证、推理参数扫描和读取已有诊断；不训练")
    log(f"候选权重：{len(models)} 个；输出目录：{OUTPUT_ROOT}")
    if missing:
        log("未找到的可选权重会跳过，完整列表见 missing_optional_model_weights.csv")
    log("=" * 76)

    run_stage1(models)
    run_stage2(models)
    run_stage3_snapshot()

    readme = [
        "NEU-DET tuning stages 1/2/3 output",
        "IMPORTANT: This script did not train any model.",
        f"data_yaml={DATA_YAML}",
        f"imgsz={IMGSZ}; val_batch={VAL_BATCH}; predict_batch={PRED_BATCH}; device={DEVICE}",
        f"stage2_conf_thresholds={CONF_THRESHOLDS}",
        f"stage2_nms_iou_thresholds={NMS_IOU_THRESHOLDS}",
        f"stage2_match_iou_thresholds={MATCH_IOU_THRESHOLDS}",
        "Thresholds are selected on the same validation split and are exploratory; do not claim independent-test generalization.",
        "Standard mAP values and fixed-threshold P/R/F1 are different evaluation views; do not mix them.",
        "In standard_validation_metrics.csv, Ultralytics P/R are at its PR-curve operating point; conf=0.001 is the low prediction floor, not the fixed operating threshold.",
        "Stage 4 box-loss training is not started by this script.",
    ]
    (OUTPUT_ROOT / "README.txt").write_text("\n".join(readme), encoding="utf-8")
    log("\n全部完成。没有启动训练。输出目录：" + str(OUTPUT_ROOT))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("\n脚本运行失败：")
        log(str(exc))
        traceback.print_exc()
        sys.exit(1)
