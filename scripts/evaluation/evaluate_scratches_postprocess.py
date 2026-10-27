# -*- coding: utf-8 -*-
"""
NEU-DET scratches：定量比较原始预测、第一轮后处理、第二轮后处理。

本脚本不调用 AI 视觉判断，也不修改模型。
它将预测框与 YOLO 格式真实标签进行一对一匹配，输出固定置信度预测下的
Precision / Recall / F1，以及每张图的误检、漏检明细。

评估范围：
- 图片：D:\\yolov8\\NEU-DET\\validation\\images\\scratches
- 标签：递归搜索 D:\\yolov8\\NEU-DET\\validation\\labels 下同名 .txt
- 基线：第一轮 detections_before.csv（已经过 YOLO 内置 NMS，尚未过自定义筛选）
- 第一轮：第一轮 detections_after.csv
- 第二轮：第二轮 detections_after.csv

注意：
1. 这是固定 conf=0.30 的预测框匹配评估，不是完整的 mAP/AP 曲线评估。
2. IoU=0.50 和 IoU=0.75 是“预测框与真实框匹配”的阈值，不是推理 NMS 参数。
3. 标签类别编号按当前 NEU-DET 配置读取；预测与真实框必须类别相同才算匹配。
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


# =========================
# 已确认的本地路径
# =========================
IMAGE_DIR = Path(r"D:\yolov8\NEU-DET\validation\images\scratches")
LABEL_ROOT = Path(r"D:\yolov8\NEU-DET\validation\labels")

ROUND1_DIR = Path(r"E:\yolo_project\runs\detect\scratches_postprocess_compare")
ROUND2_DIR = Path(r"E:\yolo_project\runs\detect\scratches_postprocess_compare_v2")

OUTPUT_DIR = Path(r"E:\yolo_project\runs\detect\scratches_evaluation")

# 评估匹配阈值。不要与推理时的 NMS IoU=0.80 混淆。
MATCH_IOU_THRESHOLDS = (0.50, 0.75)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def check_inputs() -> None:
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"找不到划痕图片目录：{IMAGE_DIR}")
    if not LABEL_ROOT.is_dir():
        raise FileNotFoundError(f"找不到标签根目录：{LABEL_ROOT}")

    required_csvs = [
        ROUND1_DIR / "detections_before.csv",
        ROUND1_DIR / "detections_after.csv",
        ROUND2_DIR / "detections_after.csv",
    ]
    missing = [str(p) for p in required_csvs if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少实验 CSV 文件，请确认第一轮、第二轮脚本均已运行：\n"
            + "\n".join(missing)
        )


def get_images() -> List[Path]:
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"图片目录中没有找到图片：{IMAGE_DIR}")
    return images


def load_ground_truths(images: List[Path]) -> Dict[str, List[Dict]]:
    """
    递归寻找与每张图片同名的标签文件，兼容 labels 根目录平铺或按子目录存放。
    YOLO 标签格式：class_id x_center y_center width height（后四项归一化）。
    """
    all_label_files = list(LABEL_ROOT.rglob("*.txt"))
    label_map: Dict[str, List[Path]] = defaultdict(list)
    for label_path in all_label_files:
        label_map[label_path.stem].append(label_path)

    missing = []
    duplicate = []
    image_sizes: Dict[str, Tuple[int, int]] = {}
    gt_by_image: Dict[str, List[Dict]] = {}

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"OpenCV 无法读取图片：{image_path}")
        height, width = image.shape[:2]
        image_sizes[image_path.name] = (width, height)

        candidates = label_map.get(image_path.stem, [])
        if not candidates:
            missing.append(image_path.name)
            continue
        if len(candidates) > 1:
            duplicate.append((image_path.name, [str(p) for p in candidates]))
            continue

        label_path = candidates[0]
        boxes: List[Dict] = []
        with label_path.open("r", encoding="utf-8-sig") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(
                        f"标签格式错误：{label_path} 第 {line_number} 行：{line}"
                    )
                class_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])

                # YOLO 归一化 xywh 转为原图像素坐标 xyxy。
                x1 = (xc - bw / 2.0) * width
                y1 = (yc - bh / 2.0) * height
                x2 = (xc + bw / 2.0) * width
                y2 = (yc + bh / 2.0) * height

                boxes.append({
                    "gt_index": len(boxes),
                    "class_id": class_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })
        gt_by_image[image_path.name] = boxes

    if missing:
        preview = "\n".join(missing[:15])
        raise FileNotFoundError(
            f"有 {len(missing)} 张图片找不到同名标签文件。示例：\n{preview}\n"
            f"请检查标签是否位于 {LABEL_ROOT} 下，并且文件名与图片一致。"
        )
    if duplicate:
        detail = "\n".join(f"{name}: {paths}" for name, paths in duplicate[:10])
        raise RuntimeError(
            "发现同名图片对应多个标签文件，无法安全判断应使用哪一个：\n" + detail
        )

    return gt_by_image


def load_predictions(csv_path: Path, images: List[Path]) -> Dict[str, List[Dict]]:
    predictions: Dict[str, List[Dict]] = {p.name: [] for p in images}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "image", "detection_index", "class_id", "class_name",
            "confidence", "x1", "y1", "x2", "y2"
        }
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"CSV 缺少必要列：{csv_path}\n缺少：{sorted(missing_columns)}"
            )

        for row in reader:
            image_name = Path(row["image"]).name
            if image_name not in predictions:
                # 忽略不属于当前 60 张图片的行。
                continue
            predictions[image_name].append({
                "pred_index": int(float(row["detection_index"])),
                "class_id": int(float(row["class_id"])),
                "class_name": row["class_name"],
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
            })

    for image_name in predictions:
        predictions[image_name].sort(
            key=lambda p: p["confidence"], reverse=True
        )
    return predictions


def calculate_iou(a: Dict, b: Dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def fmt(value: float) -> str:
    return f"{value:.6f}"


def evaluate_one_image(
    image_name: str,
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float,
    config_name: str,
) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]]:
    """
    置信度从高到低进行一对一匹配。
    每个真实框最多匹配一个预测框，每个预测框最多匹配一个真实框。
    只有类别相同且 IoU 达到阈值才算 TP。
    """
    unmatched_gt = set(range(len(ground_truths)))
    tp_rows: List[Dict] = []
    fp_rows: List[Dict] = []
    gt_best_iou = {i: 0.0 for i in range(len(ground_truths))}

    ordered_predictions = sorted(
        predictions, key=lambda p: p["confidence"], reverse=True
    )

    for pred in ordered_predictions:
        same_class_candidates = []
        best_iou_any_class = 0.0

        for gt_index, gt in enumerate(ground_truths):
            iou = calculate_iou(pred, gt)
            best_iou_any_class = max(best_iou_any_class, iou)
            if pred["class_id"] == gt["class_id"]:
                gt_best_iou[gt_index] = max(gt_best_iou[gt_index], iou)
                if gt_index in unmatched_gt:
                    same_class_candidates.append((iou, gt_index))

        if same_class_candidates:
            best_iou, best_gt_index = max(same_class_candidates, key=lambda x: x[0])
        else:
            best_iou, best_gt_index = 0.0, None

        if best_gt_index is not None and best_iou >= iou_threshold:
            unmatched_gt.remove(best_gt_index)
            gt = ground_truths[best_gt_index]
            tp_rows.append({
                "config": config_name,
                "iou_threshold": fmt(iou_threshold),
                "image": image_name,
                "pred_index": pred["pred_index"],
                "gt_index": gt["gt_index"],
                "class_id": pred["class_id"],
                "class_name": pred["class_name"],
                "confidence": fmt(pred["confidence"]),
                "matched_iou": fmt(best_iou),
                "pred_x1": fmt(pred["x1"]),
                "pred_y1": fmt(pred["y1"]),
                "pred_x2": fmt(pred["x2"]),
                "pred_y2": fmt(pred["y2"]),
                "gt_x1": fmt(gt["x1"]),
                "gt_y1": fmt(gt["y1"]),
                "gt_x2": fmt(gt["x2"]),
                "gt_y2": fmt(gt["y2"]),
            })
        else:
            fp_rows.append({
                "config": config_name,
                "iou_threshold": fmt(iou_threshold),
                "image": image_name,
                "pred_index": pred["pred_index"],
                "class_id": pred["class_id"],
                "class_name": pred["class_name"],
                "confidence": fmt(pred["confidence"]),
                "best_iou_same_class_unmatched_gt": fmt(best_iou),
                "best_iou_any_class": fmt(best_iou_any_class),
                "x1": fmt(pred["x1"]),
                "y1": fmt(pred["y1"]),
                "x2": fmt(pred["x2"]),
                "y2": fmt(pred["y2"]),
                "note": (
                    "未匹配预测；可能是误检、重复框、定位偏差或标签未覆盖，"
                    "需结合指标和少量案例判断"
                ),
            })

    fn_rows: List[Dict] = []
    for gt_index in sorted(unmatched_gt):
        gt = ground_truths[gt_index]
        fn_rows.append({
            "config": config_name,
            "iou_threshold": fmt(iou_threshold),
            "image": image_name,
            "gt_index": gt["gt_index"],
            "class_id": gt["class_id"],
            "x1": fmt(gt["x1"]),
            "y1": fmt(gt["y1"]),
            "x2": fmt(gt["x2"]),
            "y2": fmt(gt["y2"]),
            "best_iou_same_class_any_prediction": fmt(gt_best_iou[gt_index]),
            "note": "未匹配真实标注；可能是漏检或预测框定位未达到 IoU 阈值",
        })

    tp = len(tp_rows)
    fp = len(fp_rows)
    fn = len(fn_rows)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)

    per_image = {
        "config": config_name,
        "iou_threshold": fmt(iou_threshold),
        "image": image_name,
        "ground_truth_count": len(ground_truths),
        "prediction_count": len(predictions),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": fmt(precision),
        "recall": fmt(recall),
        "F1": fmt(f1),
    }
    return per_image, tp_rows, fp_rows, fn_rows


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    try:
        check_inputs()
        images = get_images()
        ground_truths = load_ground_truths(images)
    except Exception as exc:
        print(f"[输入检查失败] {exc}")
        return 1

    configs = [
        (
            "baseline_before_custom_filter",
            ROUND1_DIR / "detections_before.csv",
        ),
        (
            "round1_after_filter_0.25",
            ROUND1_DIR / "detections_after.csv",
        ),
        (
            "round2_after_filter_0.20",
            ROUND2_DIR / "detections_after.csv",
        ),
    ]

    try:
        predictions_by_config = {
            name: load_predictions(csv_path, images)
            for name, csv_path in configs
        }
    except Exception as exc:
        print(f"[预测 CSV 读取失败] {exc}")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_rows: List[Dict] = []
    per_image_rows: List[Dict] = []
    tp_rows_all: List[Dict] = []
    fp_rows_all: List[Dict] = []
    fn_rows_all: List[Dict] = []

    for config_name, _csv_path in configs:
        all_predictions = predictions_by_config[config_name]

        for iou_threshold in MATCH_IOU_THRESHOLDS:
            total_gt = 0
            total_pred = 0
            total_tp = 0
            total_fp = 0
            total_fn = 0
            per_image_for_config: List[Dict] = []

            for image_path in images:
                image_name = image_path.name
                preds = all_predictions.get(image_name, [])
                gts = ground_truths[image_name]

                per_image, tp_rows, fp_rows, fn_rows = evaluate_one_image(
                    image_name=image_name,
                    predictions=preds,
                    ground_truths=gts,
                    iou_threshold=iou_threshold,
                    config_name=config_name,
                )
                per_image_rows.append(per_image)
                per_image_for_config.append(per_image)
                tp_rows_all.extend(tp_rows)
                fp_rows_all.extend(fp_rows)
                fn_rows_all.extend(fn_rows)

                total_gt += len(gts)
                total_pred += len(preds)
                total_tp += per_image["TP"]
                total_fp += per_image["FP"]
                total_fn += per_image["FN"]

            precision = safe_ratio(total_tp, total_tp + total_fp)
            recall = safe_ratio(total_tp, total_tp + total_fn)
            f1 = safe_ratio(2 * precision * recall, precision + recall)

            overall_rows.append({
                "config": config_name,
                "match_iou_threshold": fmt(iou_threshold),
                "images": len(images),
                "ground_truth_count": total_gt,
                "prediction_count": total_pred,
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "precision": fmt(precision),
                "recall": fmt(recall),
                "F1": fmt(f1),
                "prediction_source_csv": str(_csv_path),
            })

    write_csv(
        OUTPUT_DIR / "overall_metrics.csv",
        overall_rows,
        [
            "config", "match_iou_threshold", "images", "ground_truth_count",
            "prediction_count", "TP", "FP", "FN", "precision", "recall",
            "F1", "prediction_source_csv",
        ],
    )
    write_csv(
        OUTPUT_DIR / "per_image_metrics.csv",
        per_image_rows,
        [
            "config", "iou_threshold", "image", "ground_truth_count",
            "prediction_count", "TP", "FP", "FN", "precision", "recall", "F1",
        ],
    )
    write_csv(
        OUTPUT_DIR / "matched_detections.csv",
        tp_rows_all,
        [
            "config", "iou_threshold", "image", "pred_index", "gt_index",
            "class_id", "class_name", "confidence", "matched_iou",
            "pred_x1", "pred_y1", "pred_x2", "pred_y2",
            "gt_x1", "gt_y1", "gt_x2", "gt_y2",
        ],
    )
    write_csv(
        OUTPUT_DIR / "false_positive_predictions.csv",
        fp_rows_all,
        [
            "config", "iou_threshold", "image", "pred_index", "class_id",
            "class_name", "confidence", "best_iou_same_class_unmatched_gt",
            "best_iou_any_class", "x1", "y1", "x2", "y2", "note",
        ],
    )
    write_csv(
        OUTPUT_DIR / "missed_ground_truth.csv",
        fn_rows_all,
        [
            "config", "iou_threshold", "image", "gt_index", "class_id",
            "x1", "y1", "x2", "y2",
            "best_iou_same_class_any_prediction", "note",
        ],
    )

    notes = [
        "NEU-DET scratches 定量评估说明",
        f"图片目录：{IMAGE_DIR}",
        f"标签根目录：{LABEL_ROOT}",
        f"输出目录：{OUTPUT_DIR}",
        f"图片数量：{len(images)}",
        f"真实标注目标总数：{sum(len(v) for v in ground_truths.values())}",
        "baseline_before_custom_filter = YOLO 内置 NMS 后、自定义筛选前的预测。",
        "round1_after_filter_0.25 = 第一轮自定义后处理结果。",
        "round2_after_filter_0.20 = 第二轮自定义后处理结果。",
        "所有指标均基于现有固定置信度 conf=0.30 的预测 CSV。",
        "match_iou_threshold 是预测框与真实框的一对一匹配阈值，不是 NMS IoU。",
        "这些是固定预测阈值下的 Precision/Recall/F1，不等同于 mAP/AP。",
        "未匹配预测不一定全是误检：也可能是重复框、定位偏差或标签未覆盖。",
        "未匹配真实框不一定全是模型完全没检测到：也可能是预测框 IoU 未达到匹配阈值。",
    ]
    (OUTPUT_DIR / "README.txt").write_text("\n".join(notes), encoding="utf-8")

    print("\n" + "=" * 88)
    print("划痕检测定量评估完成")
    print(f"图片数：{len(images)}")
    print(f"真实标注目标数：{sum(len(v) for v in ground_truths.values())}")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 88)
    print(
        f"{'配置':36s} {'IoU':>5s} {'GT':>5s} {'Pred':>6s} "
        f"{'TP':>5s} {'FP':>5s} {'FN':>5s} {'P':>8s} {'R':>8s} {'F1':>8s}"
    )
    for row in overall_rows:
        if row["match_iou_threshold"] == fmt(0.50):
            print(
                f"{row['config']:36s} "
                f"{float(row['match_iou_threshold']):5.2f} "
                f"{int(row['ground_truth_count']):5d} "
                f"{int(row['prediction_count']):6d} "
                f"{int(row['TP']):5d} {int(row['FP']):5d} {int(row['FN']):5d} "
                f"{float(row['precision']):8.4f} "
                f"{float(row['recall']):8.4f} "
                f"{float(row['F1']):8.4f}"
            )
    print("\n生成文件：")
    print(f"  {OUTPUT_DIR / 'overall_metrics.csv'}")
    print(f"  {OUTPUT_DIR / 'per_image_metrics.csv'}")
    print(f"  {OUTPUT_DIR / 'false_positive_predictions.csv'}")
    print(f"  {OUTPUT_DIR / 'missed_ground_truth.csv'}")
    print(f"  {OUTPUT_DIR / 'matched_detections.csv'}")
    print(f"  {OUTPUT_DIR / 'README.txt'}")
    print("\n请先把 overall_metrics.csv 发回，再根据结果决定下一步。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
