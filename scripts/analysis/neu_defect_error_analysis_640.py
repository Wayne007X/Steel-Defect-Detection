# -*- coding: utf-8 -*-
"""
NEU-DET 六类钢材表面缺陷：自动化错误分析（不训练模型）
适用于 Windows + Cursor PowerShell，不使用 Cursor Agent。

比较模型：
1) E:\\yolo_project\\runs\\neu_defect_clean\\weights\\best.pt
2) E:\\yolo_project\\runs\\neu_defect_mosaic000\\weights\\best.pt

统一设置：
- imgsz=640
- 预测候选置信度下限=0.001
- NMS IoU=0.70
- 固定统计工作点 conf=0.25
- GT/预测匹配 IoU=0.50

输出：
- per_class_summary.csv：两模型 × 六类别 + all 的 TP/FP/FN、Precision/Recall 和错误类型计数
- gt_match_details.csv：每个真实标注的匹配状态、候选框与漏检类型
- prediction_details.csv：每个预测框的匹配状态与误检类型

注意：
本脚本统计的是 conf=0.25、IoU=0.50 下的自定义逐图匹配结果，不是 mAP。
mAP 仍应参考 Ultralytics 的标准验证结果。
"""

from pathlib import Path
from collections import defaultdict
import csv
import math
import os
import sys
import traceback

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# -------------------- 已确认的本地路径 --------------------
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
IMAGE_ROOT = Path(r"D:\yolov8\NEU-DET\validation\images")
LABEL_ROOT = Path(r"D:\yolov8\NEU-DET\validation\labels")

MODELS = [
    (
        "neu_defect_clean",
        Path(r"E:\yolo_project\runs\neu_defect_clean\weights\best.pt"),
    ),
    (
        "neu_defect_mosaic000",
        Path(r"E:\yolo_project\runs\neu_defect_mosaic000\weights\best.pt"),
    ),
]

OUTPUT_DIR = Path(r"E:\yolo_project\runs\neu_defect_error_analysis_640")

# -------------------- 统一评估参数 --------------------
IMGSZ = 640
PRED_CONF_MIN = 0.001       # 保留低置信度候选，用于诊断低置信度漏检
NMS_IOU = 0.70
OPERATING_CONF = 0.25       # 固定工作点，只用于 TP/FP/FN 统计
MATCH_IOU = 0.50             # 正确检测的 IoU 匹配阈值
DIAGNOSTIC_IOU = 0.10        # 低于 MATCH_IOU 的部分重叠仅作为错误线索
MAX_DET = 300
PRED_BATCH = 8
DEVICE = "0"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def norm_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def get_class_names(data_yaml: Path):
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    names = data.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    raise ValueError(f"无法从 data.yaml 读取 names：{data_yaml}")


def collect_images(image_root: Path):
    paths = sorted(
        [p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda p: str(p.relative_to(image_root)).lower(),
    )
    if not paths:
        raise FileNotFoundError(f"没有找到验证图片：{image_root}")
    return paths


def find_label_path(image_path: Path) -> Path:
    """优先匹配 labels/<相对图片路径>.txt，兼容 labels/<文件名>.txt 的平铺布局。"""
    rel = image_path.relative_to(IMAGE_ROOT).with_suffix(".txt")
    mirrored = LABEL_ROOT / rel
    if mirrored.is_file():
        return mirrored

    flat = LABEL_ROOT / image_path.with_suffix(".txt").name
    if flat.is_file():
        return flat

    # 最后尝试按同名标签在 labels 下递归搜索；若重复则不擅自猜测。
    candidates = list(LABEL_ROOT.rglob(image_path.with_suffix(".txt").name))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"标签文件存在多个同名候选，无法安全匹配：{image_path.name}\n"
            + "\n".join(str(p) for p in candidates)
        )
    raise FileNotFoundError(f"找不到图片对应的标签：{image_path}\n尝试路径：{mirrored}")


def read_ground_truth(image_path: Path, label_path: Path, class_names):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取图片：{image_path}")
    height, width = image.shape[:2]
    objects = []

    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"标签格式错误：{label_path}:{line_no} -> {line}")

        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])

        if cls_id < 0 or cls_id >= len(class_names):
            raise ValueError(
                f"类别 ID 超出 data.yaml names 范围：{label_path}:{line_no}，class={cls_id}"
            )

        # YOLO 标签为归一化 xywh，转换为原图像素坐标 xyxy
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        objects.append({
            "class_id": cls_id,
            "class_name": class_names[cls_id],
            "xyxy": [x1, y1, x2, y2],
        })

    return objects, width, height


def box_iou_matrix(boxes_a, boxes_b):
    """返回 A×B 的 IoU 矩阵。输入为 xyxy 列表。"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)

    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(right_bottom - left_top, 0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def best_candidate(preds, ious, gt_index, class_id=None, min_conf=None, excluded=None):
    """返回符合条件的最佳预测索引；优先 IoU，其次置信度。"""
    excluded = excluded or set()
    options = []
    for pi, pred in enumerate(preds):
        if pi in excluded:
            continue
        if class_id is not None and pred["class_id"] != class_id:
            continue
        if min_conf is not None and pred["conf"] < min_conf:
            continue
        iou = float(ious[gt_index, pi]) if ious.size else 0.0
        options.append((iou, pred["conf"], pi))
    if not options:
        return None
    options.sort(reverse=True)
    return options[0][2]


def classify_missed_gt(gt_index, gt, preds, ious, matched_pred_for_gt, pred_to_gt):
    """对未匹配 GT 给出互斥的主要错误类型。"""
    same_any = best_candidate(preds, ious, gt_index, class_id=gt["class_id"])
    same_high = best_candidate(
        preds, ious, gt_index, class_id=gt["class_id"], min_conf=OPERATING_CONF
    )
    same_low = best_candidate(
        preds, ious, gt_index, class_id=gt["class_id"], min_conf=None
    )

    # 先判断同类高置信度候选是否已被分配给另一个 GT。
    if same_high is not None and float(ious[gt_index, same_high]) >= MATCH_IOU:
        if same_high in pred_to_gt:
            return "candidate_assigned_to_other_gt", same_high

    # 同类别候选达到匹配 IoU，但分数低于工作点。
    if same_low is not None and float(ious[gt_index, same_low]) >= MATCH_IOU:
        if preds[same_low]["conf"] < OPERATING_CONF:
            return "low_confidence_same_class", same_low

    # 其他类别的高置信度预测覆盖了真实目标。
    other_options = [
        pi for pi, p in enumerate(preds)
        if p["class_id"] != gt["class_id"] and p["conf"] >= OPERATING_CONF
        and float(ious[gt_index, pi]) >= MATCH_IOU
    ]
    if other_options:
        other_options.sort(
            key=lambda pi: (float(ious[gt_index, pi]), preds[pi]["conf"]),
            reverse=True,
        )
        return "class_confusion", other_options[0]

    # 同类预测有重叠但定位未达到 0.5。
    same_partial = [
        pi for pi, p in enumerate(preds)
        if p["class_id"] == gt["class_id"]
        and float(ious[gt_index, pi]) >= DIAGNOSTIC_IOU
    ]
    if same_partial:
        same_partial.sort(
            key=lambda pi: (float(ious[gt_index, pi]), preds[pi]["conf"]),
            reverse=True,
        )
        return "localization_or_box_extent", same_partial[0]

    # 其他类别预测只有部分重叠，作为线索而非确定的类别错误。
    any_partial = [
        pi for pi in range(len(preds))
        if float(ious[gt_index, pi]) >= DIAGNOSTIC_IOU
    ]
    if any_partial:
        any_partial.sort(
            key=lambda pi: (float(ious[gt_index, pi]), preds[pi]["conf"]),
            reverse=True,
        )
        return "partial_overlap_other_candidate", any_partial[0]

    return "no_candidate_iou_ge_0.10", None


def classify_false_positive(pred_index, pred, gts, ious, pred_to_gt):
    """给未匹配的高置信度预测归类。"""
    if not gts:
        return "background_fp", None, 0.0

    overlaps = [(float(ious[gi, pred_index]), gi) for gi in range(len(gts))]
    overlaps.sort(reverse=True)
    best_iou, best_gt_index = overlaps[0]
    same = [(iou, gi) for iou, gi in overlaps if gts[gi]["class_id"] == pred["class_id"]]
    other = [(iou, gi) for iou, gi in overlaps if gts[gi]["class_id"] != pred["class_id"]]

    if same and same[0][0] >= MATCH_IOU:
        # 同类 GT 已经被另一预测匹配，当前预测多半是重复框。
        return "duplicate_or_competing_prediction", same[0][1], same[0][0]

    if other and other[0][0] >= MATCH_IOU:
        return "class_confusion_fp", other[0][1], other[0][0]

    if same and same[0][0] >= DIAGNOSTIC_IOU:
        return "localization_fp", same[0][1], same[0][0]

    if best_iou >= DIAGNOSTIC_IOU:
        return "partial_overlap_other_class_fp", best_gt_index, best_iou

    return "background_fp", best_gt_index, best_iou


def predict_model(model_name, model_path, image_paths, labels_by_path, class_names):
    print("\n" + "=" * 76)
    print(f"开始分析模型：{model_name}")
    print(f"权重：{model_path}")
    print(f"图片数：{len(image_paths)}，imgsz={IMGSZ}，预测 conf 下限={PRED_CONF_MIN}")
    print("=" * 76)

    model = YOLO(str(model_path))
    image_sources = [str(p) for p in image_paths]
    pred_results = model.predict(
        source=image_sources,
        imgsz=IMGSZ,
        conf=PRED_CONF_MIN,
        iou=NMS_IOU,
        max_det=MAX_DET,
        device=DEVICE,
        batch=PRED_BATCH,
        stream=True,
        verbose=False,
        save=False,
        augment=False,
    )

    image_data = {}
    for index, (image_path, result) in enumerate(zip(image_paths, pred_results), start=1):
        gts = labels_by_path[norm_path(image_path)]
        preds = []
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            for j, (box, conf, cls_id) in enumerate(zip(xyxy, confs, classes)):
                if cls_id < 0 or cls_id >= len(class_names):
                    continue
                preds.append({
                    "pred_index": j,
                    "class_id": int(cls_id),
                    "class_name": class_names[int(cls_id)],
                    "conf": float(conf),
                    "xyxy": [float(v) for v in box.tolist()],
                })

        image_data[norm_path(image_path)] = {
            "image_path": image_path,
            "gts": gts,
            "preds": preds,
        }

        if index % 30 == 0 or index == len(image_paths):
            print(f"[{index:03d}/{len(image_paths)}] 已完成")

    del model
    return image_data


def analyze_one_model(model_name, image_data, class_names):
    # 初始化每类计数
    counts = {}
    for cls_id, cls_name in enumerate(class_names):
        counts[cls_id] = {
            "model": model_name,
            "class_id": cls_id,
            "class_name": cls_name,
            "gt_count": 0,
            "pred_count_at_conf": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "low_confidence_fn": 0,
            "class_confusion_fn": 0,
            "localization_fn": 0,
            "no_candidate_fn": 0,
            "candidate_assigned_to_other_gt_fn": 0,
            "partial_overlap_other_candidate_fn": 0,
            "duplicate_or_competing_prediction_fp": 0,
            "class_confusion_fp": 0,
            "localization_fp": 0,
            "partial_overlap_other_class_fp": 0,
            "background_fp": 0,
        }

    gt_detail_rows = []
    pred_detail_rows = []

    for image_key, record in image_data.items():
        image_path = record["image_path"]
        gts = record["gts"]
        preds = record["preds"]

        for gt in gts:
            counts[gt["class_id"]]["gt_count"] += 1

        for pred in preds:
            if pred["conf"] >= OPERATING_CONF:
                counts[pred["class_id"]]["pred_count_at_conf"] += 1

        ious = box_iou_matrix(
            [gt["xyxy"] for gt in gts],
            [pred["xyxy"] for pred in preds],
        )

        # 按置信度从高到低进行一对一匹配：只将同类别且 IoU>=0.5 计为 TP。
        eligible = [
            pi for pi, p in enumerate(preds)
            if p["conf"] >= OPERATING_CONF
        ]
        eligible.sort(key=lambda pi: preds[pi]["conf"], reverse=True)

        used_gts = set()
        pred_to_gt = {}
        gt_to_pred = {}

        for pi in eligible:
            pred = preds[pi]
            candidates = []
            for gi, gt in enumerate(gts):
                if gi in used_gts or gt["class_id"] != pred["class_id"]:
                    continue
                iou = float(ious[gi, pi])
                if iou >= MATCH_IOU:
                    candidates.append((iou, gi))
            if candidates:
                candidates.sort(reverse=True)
                best_iou, gi = candidates[0]
                used_gts.add(gi)
                pred_to_gt[pi] = gi
                gt_to_pred[gi] = pi
                counts[pred["class_id"]]["tp"] += 1

        # 真实目标详情：每个 GT 都有一行。
        for gi, gt in enumerate(gts):
            matched_pi = gt_to_pred.get(gi)
            if matched_pi is not None:
                matched_pred = preds[matched_pi]
                error_type = "TP"
                matched_iou = float(ious[gi, matched_pi])
                best_same_pi = best_candidate(preds, ious, gi, class_id=gt["class_id"])
                best_any_pi = best_candidate(preds, ious, gi)
            else:
                counts[gt["class_id"]]["fn"] += 1
                error_type, diagnostic_pi = classify_missed_gt(
                    gi, gt, preds, ious, gt_to_pred, pred_to_gt
                )
                if error_type == "low_confidence_same_class":
                    counts[gt["class_id"]]["low_confidence_fn"] += 1
                elif error_type == "class_confusion":
                    counts[gt["class_id"]]["class_confusion_fn"] += 1
                elif error_type == "localization_or_box_extent":
                    counts[gt["class_id"]]["localization_fn"] += 1
                elif error_type == "no_candidate_iou_ge_0.10":
                    counts[gt["class_id"]]["no_candidate_fn"] += 1
                elif error_type == "candidate_assigned_to_other_gt":
                    counts[gt["class_id"]]["candidate_assigned_to_other_gt_fn"] += 1
                elif error_type == "partial_overlap_other_candidate":
                    counts[gt["class_id"]]["partial_overlap_other_candidate_fn"] += 1

                matched_pi = diagnostic_pi
                matched_pred = preds[matched_pi] if matched_pi is not None else None
                matched_iou = float(ious[gi, matched_pi]) if matched_pi is not None else 0.0
                best_same_pi = best_candidate(preds, ious, gi, class_id=gt["class_id"])
                best_any_pi = best_candidate(preds, ious, gi)

            gt_detail_rows.append({
                "model": model_name,
                "image": str(image_path),
                "gt_index": gi,
                "gt_class_id": gt["class_id"],
                "gt_class": gt["class_name"],
                "gt_x1": round(gt["xyxy"][0], 3),
                "gt_y1": round(gt["xyxy"][1], 3),
                "gt_x2": round(gt["xyxy"][2], 3),
                "gt_y2": round(gt["xyxy"][3], 3),
                "status": "TP" if gi in gt_to_pred else "FN",
                "error_type": error_type,
                "matched_or_diagnostic_pred_index": (
                    preds[matched_pi]["pred_index"] if matched_pi is not None else ""
                ),
                "matched_or_diagnostic_pred_class": (
                    matched_pred["class_name"] if matched_pred is not None else ""
                ),
                "matched_or_diagnostic_pred_conf": (
                    round(matched_pred["conf"], 5) if matched_pred is not None else ""
                ),
                "matched_or_diagnostic_iou": round(matched_iou, 5),
                "best_same_class_conf": (
                    round(preds[best_same_pi]["conf"], 5) if best_same_pi is not None else ""
                ),
                "best_same_class_iou": (
                    round(float(ious[gi, best_same_pi]), 5) if best_same_pi is not None else ""
                ),
                "best_any_class": (
                    preds[best_any_pi]["class_name"] if best_any_pi is not None else ""
                ),
                "best_any_conf": (
                    round(preds[best_any_pi]["conf"], 5) if best_any_pi is not None else ""
                ),
                "best_any_iou": (
                    round(float(ious[gi, best_any_pi]), 5) if best_any_pi is not None else ""
                ),
            })

        # 预测详情：TP 和所有高置信度 FP 都记录；低于工作点的候选也保留用于诊断。
        for pi, pred in enumerate(preds):
            is_tp = pi in pred_to_gt
            matched_gt_index = pred_to_gt.get(pi)
            if is_tp:
                fp_type = ""
                best_gt_index = matched_gt_index
                best_gt_iou = float(ious[matched_gt_index, pi])
            elif pred["conf"] >= OPERATING_CONF:
                counts[pred["class_id"]]["fp"] += 1
                fp_type, best_gt_index, best_gt_iou = classify_false_positive(
                    pi, pred, gts, ious, pred_to_gt
                )
                counts[pred["class_id"]][fp_type] += 1
            else:
                fp_type = "below_operating_conf_diagnostic_only"
                if len(gts):
                    overlaps = [(float(ious[gi, pi]), gi) for gi in range(len(gts))]
                    best_gt_iou, best_gt_index = max(overlaps)
                else:
                    best_gt_index, best_gt_iou = None, 0.0

            best_gt = gts[best_gt_index] if best_gt_index is not None and gts else None
            pred_detail_rows.append({
                "model": model_name,
                "image": str(image_path),
                "pred_index": pred["pred_index"],
                "pred_class_id": pred["class_id"],
                "pred_class": pred["class_name"],
                "confidence": round(pred["conf"], 6),
                "above_operating_conf": pred["conf"] >= OPERATING_CONF,
                "x1": round(pred["xyxy"][0], 3),
                "y1": round(pred["xyxy"][1], 3),
                "x2": round(pred["xyxy"][2], 3),
                "y2": round(pred["xyxy"][3], 3),
                "status": "TP" if is_tp else ("FP" if pred["conf"] >= OPERATING_CONF else "diagnostic_only"),
                "fp_type": fp_type,
                "matched_gt_index": matched_gt_index if matched_gt_index is not None else "",
                "matched_gt_class": (
                    gts[matched_gt_index]["class_name"] if matched_gt_index is not None else ""
                ),
                "best_overlap_gt_index": best_gt_index if best_gt_index is not None else "",
                "best_overlap_gt_class": best_gt["class_name"] if best_gt is not None else "",
                "best_overlap_iou": round(float(best_gt_iou), 5),
            })

    # 生成每类和 all 的 summary 行
    summary_rows = []
    metric_keys = [
        "gt_count", "pred_count_at_conf", "tp", "fp", "fn",
        "low_confidence_fn", "class_confusion_fn", "localization_fn",
        "no_candidate_fn", "candidate_assigned_to_other_gt_fn",
        "partial_overlap_other_candidate_fn",
        "duplicate_or_competing_prediction_fp", "class_confusion_fp",
        "localization_fp", "partial_overlap_other_class_fp", "background_fp",
    ]

    for cls_id, cls_name in enumerate(class_names):
        row = dict(counts[cls_id])
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        row["recall"] = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        summary_rows.append(row)

    all_row = {
        "model": model_name,
        "class_id": "",
        "class_name": "all",
    }
    for key in metric_keys:
        all_row[key] = sum(row[key] for row in counts.values())
    all_row["precision"] = (
        all_row["tp"] / (all_row["tp"] + all_row["fp"])
        if all_row["tp"] + all_row["fp"] else 0.0
    )
    all_row["recall"] = (
        all_row["tp"] / (all_row["tp"] + all_row["fn"])
        if all_row["tp"] + all_row["fn"] else 0.0
    )
    summary_rows.append(all_row)

    return summary_rows, gt_detail_rows, pred_detail_rows


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    for required in (DATA_YAML, IMAGE_ROOT, LABEL_ROOT):
        if not required.exists():
            raise FileNotFoundError(f"找不到所需路径：{required}")
    for model_name, model_path in MODELS:
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到模型权重（{model_name}）：{model_path}")

    class_names = get_class_names(DATA_YAML)
    image_paths = collect_images(IMAGE_ROOT)
    print(f"验证图片：{len(image_paths)} 张")
    print(f"类别：{class_names}")
    print(f"输出目录：{OUTPUT_DIR}")

    # 先检查所有标签都能找到并解析，避免跑完模型才发现路径错误。
    labels_by_path = {}
    missing_label_errors = []
    for image_path in image_paths:
        try:
            label_path = find_label_path(image_path)
            gts, _, _ = read_ground_truth(image_path, label_path, class_names)
            labels_by_path[norm_path(image_path)] = gts
        except Exception as exc:
            missing_label_errors.append(str(exc))

    if missing_label_errors:
        print("\n标签检查失败，停止分析以免产生误导性指标。前 20 个问题：")
        for error in missing_label_errors[:20]:
            print(f"- {error}")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary = []
    all_gt_details = []
    all_pred_details = []

    for model_name, model_path in MODELS:
        image_data = predict_model(
            model_name, model_path, image_paths, labels_by_path, class_names
        )
        summary, gt_details, pred_details = analyze_one_model(
            model_name, image_data, class_names
        )
        all_summary.extend(summary)
        all_gt_details.extend(gt_details)
        all_pred_details.extend(pred_details)

        all_row = next(r for r in summary if r["class_name"] == "all")
        print(
            f"\n{model_name}: TP={all_row['tp']}, FP={all_row['fp']}, FN={all_row['fn']}, "
            f"P={all_row['precision']:.3f}, R={all_row['recall']:.3f}"
        )
        for row in summary:
            if row["class_name"] in ("crazing", "rolled-in_scale"):
                print(
                    f"  {row['class_name']}: TP={row['tp']} FP={row['fp']} FN={row['fn']} "
                    f"P={row['precision']:.3f} R={row['recall']:.3f}; "
                    f"低置信度漏检={row['low_confidence_fn']}, "
                    f"类别混淆漏检={row['class_confusion_fn']}, "
                    f"定位问题漏检={row['localization_fn']}, "
                    f"无重叠候选={row['no_candidate_fn']}"
                )

    write_csv(OUTPUT_DIR / "per_class_summary.csv", all_summary)
    write_csv(OUTPUT_DIR / "gt_match_details.csv", all_gt_details)
    write_csv(OUTPUT_DIR / "prediction_details.csv", all_pred_details)

    print("\n" + "=" * 76)
    print("错误分析完成（没有进行训练）。")
    print(f"类别汇总：{OUTPUT_DIR / 'per_class_summary.csv'}")
    print(f"真实标注匹配明细：{OUTPUT_DIR / 'gt_match_details.csv'}")
    print(f"预测框明细：{OUTPUT_DIR / 'prediction_details.csv'}")
    print(f"固定工作点：conf={OPERATING_CONF}, match IoU={MATCH_IOU}")
    print("注意：上述 TP/FP/FN 是本脚本的逐图贪心一对一匹配统计，不是 Ultralytics mAP。")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n脚本运行失败：")
        print(exc)
        traceback.print_exc()
        sys.exit(1)
