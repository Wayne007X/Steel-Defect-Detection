# -*- coding: utf-8 -*-
"""
NEU-DET scratches：置信度阈值扫描（不使用 Cursor Agent）。

用途：
1. 用当前 best.pt 对 60 张 scratches 图片推理一次，最低 conf=0.05。
2. 在同一批候选预测上扫描 conf=0.05, 0.10, ..., 0.70。
3. 比较三种方案：
   - baseline：仅 YOLO 内置 NMS
   - filter_0.25：自定义去重，ELONGATED_PERP_CENTER_MAX=0.25
   - filter_0.20：自定义去重，ELONGATED_PERP_CENTER_MAX=0.20
4. 计算 IoU 匹配阈值 0.50 和 0.75 下的 TP/FP/FN/Precision/Recall/F1。

重要：
- MATCH_IOU 是预测框与真实标注的一对一匹配阈值，不是 NMS IoU。
- 固定 conf=0.05 推理，再按置信度筛选，可比较 0.05 以上的阈值。
- 这不是完整 mAP/AP 曲线评估。
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


# 已确认路径
MODEL_PATH = r"E:\yolo_project\runs\neu_defect_clean\weights\best.pt"
IMAGE_DIR = Path(r"D:\yolov8\NEU-DET\validation\images\scratches")
LABEL_ROOT = Path(r"D:\yolov8\NEU-DET\validation\labels")
OUTPUT_DIR = Path(r"E:\yolo_project\runs\detect\scratches_conf_sweep")

# 推理设置
IMGSZ = 640
INFERENCE_CONF = 0.05
NMS_IOU = 0.80
MAX_DET = 1000

# 评估设置
CONF_THRESHOLDS = [round(i / 100, 2) for i in range(5, 71, 5)]
MATCH_IOUS = (0.50, 0.75)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 与前两轮相同的其他后处理规则
ELONGATED_ASPECT_RATIO = 2.2
ELONGATED_LONG_OVERLAP_MIN = 0.80
ELONGATED_LONG_CENTER_MAX = 0.60
ELONGATED_AREA_RATIO_MIN = 0.35
ELONGATED_LONG_SIZE_RATIO_MIN = 0.40
ELONGATED_THICKNESS_RATIO_MIN = 0.50
GENERAL_IOU_MIN = 0.82
GENERAL_CENTER_DIST_MAX = 0.25
GENERAL_WIDTH_RATIO_MIN = 0.55
GENERAL_HEIGHT_RATIO_MIN = 0.55
CONTAINMENT_MIN = 0.94
CONTAINMENT_CENTER_DIST_MAX = 0.18
CONTAINMENT_AREA_RATIO_MIN = 0.60
CONTAINMENT_WIDTH_RATIO_MIN = 0.55
CONTAINMENT_HEIGHT_RATIO_MIN = 0.55


def require_inputs() -> None:
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(f"找不到模型权重：{MODEL_PATH}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"找不到图片目录：{IMAGE_DIR}")
    if not LABEL_ROOT.is_dir():
        raise FileNotFoundError(f"找不到标签根目录：{LABEL_ROOT}")


def get_images() -> List[Path]:
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"没有找到图片：{IMAGE_DIR}")
    return images


def load_ground_truths(images: List[Path]) -> Dict[str, List[Dict]]:
    label_by_stem: Dict[str, List[Path]] = defaultdict(list)
    for path in LABEL_ROOT.rglob("*.txt"):
        label_by_stem[path.stem].append(path)

    gts: Dict[str, List[Dict]] = {}
    missing, duplicate = [], []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"无法读取图片：{image_path}")
        h, w = image.shape[:2]

        matches = label_by_stem.get(image_path.stem, [])
        if not matches:
            missing.append(image_path.name)
            continue
        if len(matches) > 1:
            duplicate.append((image_path.name, [str(x) for x in matches]))
            continue

        boxes = []
        with matches[0].open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"标签格式错误：{matches[0]} 第 {line_no} 行")
                cls = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                boxes.append({
                    "gt_index": len(boxes),
                    "class_id": cls,
                    "x1": (xc - bw / 2) * w,
                    "y1": (yc - bh / 2) * h,
                    "x2": (xc + bw / 2) * w,
                    "y2": (yc + bh / 2) * h,
                })
        gts[image_path.name] = boxes

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} 张图片没有找到标签。示例：{', '.join(missing[:10])}"
        )
    if duplicate:
        raise RuntimeError(
            "发现同名图片对应多个标签文件，无法确定正确标签：\n"
            + "\n".join(f"{n}: {ps}" for n, ps in duplicate[:10])
        )
    return gts


def box_geometry(a: Dict, b: Dict) -> Dict[str, float]:
    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]
    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    area_a, area_b = aw * ah, bw * bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    ios = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    return {
        "aw": aw, "ah": ah, "bw": bw, "bh": bh,
        "area_a": area_a, "area_b": area_b,
        "iou": iou, "ios": ios,
        "acx": acx, "acy": acy, "bcx": bcx, "bcy": bcy,
        "center_dist": math.hypot(acx - bcx, acy - bcy),
        "min_diag": max(1.0, min(math.hypot(aw, ah), math.hypot(bw, bh))),
        "width_ratio": min(aw, bw) / max(aw, bw),
        "height_ratio": min(ah, bh) / max(ah, bh),
        "area_ratio": min(area_a, area_b) / max(area_a, area_b),
        "inter_w": inter_w, "inter_h": inter_h,
    }


def duplicate_reason(a: Dict, b: Dict, perp_max: float) -> Optional[str]:
    if a["class_id"] != b["class_id"]:
        return None
    g = box_geometry(a, b)
    aw, ah, bw, bh = g["aw"], g["ah"], g["bw"], g["bh"]

    a_h, b_h = aw / ah >= ELONGATED_ASPECT_RATIO, bw / bh >= ELONGATED_ASPECT_RATIO
    a_v, b_v = ah / aw >= ELONGATED_ASPECT_RATIO, bh / bw >= ELONGATED_ASPECT_RATIO

    if a_h and b_h:
        perp = abs(g["acy"] - g["bcy"])
        min_thick = max(1.0, min(ah, bh))
        long_center = abs(g["acx"] - g["bcx"])
        min_long = max(1.0, min(aw, bw))
        overlap = g["inter_w"] / min_long
        if perp > perp_max * min_thick:
            return None
        if (
            overlap >= ELONGATED_LONG_OVERLAP_MIN
            and long_center <= ELONGATED_LONG_CENTER_MAX * min_long
            and g["area_ratio"] >= ELONGATED_AREA_RATIO_MIN
            and g["width_ratio"] >= ELONGATED_LONG_SIZE_RATIO_MIN
            and g["height_ratio"] >= ELONGATED_THICKNESS_RATIO_MIN
        ):
            return "elongated_horizontal_duplicate"
        return None

    if a_v and b_v:
        perp = abs(g["acx"] - g["bcx"])
        min_thick = max(1.0, min(aw, bw))
        long_center = abs(g["acy"] - g["bcy"])
        min_long = max(1.0, min(ah, bh))
        overlap = g["inter_h"] / min_long
        if perp > perp_max * min_thick:
            return None
        if (
            overlap >= ELONGATED_LONG_OVERLAP_MIN
            and long_center <= ELONGATED_LONG_CENTER_MAX * min_long
            and g["area_ratio"] >= ELONGATED_AREA_RATIO_MIN
            and g["height_ratio"] >= ELONGATED_LONG_SIZE_RATIO_MIN
            and g["width_ratio"] >= ELONGATED_THICKNESS_RATIO_MIN
        ):
            return "elongated_vertical_duplicate"
        return None

    if (
        g["iou"] >= GENERAL_IOU_MIN
        and g["center_dist"] <= GENERAL_CENTER_DIST_MAX * g["min_diag"]
        and g["width_ratio"] >= GENERAL_WIDTH_RATIO_MIN
        and g["height_ratio"] >= GENERAL_HEIGHT_RATIO_MIN
    ):
        return "high_iou_similar_size"

    if (
        g["ios"] >= CONTAINMENT_MIN
        and g["center_dist"] <= CONTAINMENT_CENTER_DIST_MAX * g["min_diag"]
        and g["area_ratio"] >= CONTAINMENT_AREA_RATIO_MIN
        and g["width_ratio"] >= CONTAINMENT_WIDTH_RATIO_MIN
        and g["height_ratio"] >= CONTAINMENT_HEIGHT_RATIO_MIN
    ):
        return "near_complete_containment"

    return None


def custom_filter(dets: List[Dict], perp_max: float) -> List[Dict]:
    ordered = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict] = []
    for candidate in ordered:
        duplicate = False
        for keeper in kept:
            if duplicate_reason(keeper, candidate, perp_max):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def iou_xyxy(a: Dict, b: Dict) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(pred_by_image: Dict[str, List[Dict]], gt_by_image: Dict[str, List[Dict]], match_iou: float):
    tp = fp = fn = pred_total = gt_total = 0
    for image_name, gts in gt_by_image.items():
        preds = sorted(pred_by_image.get(image_name, []), key=lambda x: x["confidence"], reverse=True)
        unmatched = set(range(len(gts)))
        gt_total += len(gts)
        pred_total += len(preds)

        for pred in preds:
            candidates = []
            for gi in unmatched:
                gt = gts[gi]
                if pred["class_id"] == gt["class_id"]:
                    candidates.append((iou_xyxy(pred, gt), gi))
            if candidates:
                best_iou, best_gi = max(candidates, key=lambda x: x[0])
            else:
                best_iou, best_gi = 0.0, None
            if best_gi is not None and best_iou >= match_iou:
                tp += 1
                unmatched.remove(best_gi)
            else:
                fp += 1
        fn += len(unmatched)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": len(gt_by_image),
        "ground_truth_count": gt_total,
        "prediction_count": pred_total,
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "F1": f1,
    }


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    try:
        require_inputs()
        images = get_images()
        gt_by_image = load_ground_truths(images)
    except Exception as e:
        print(f"[输入检查失败] {e}")
        return 1

    print("=" * 76)
    print("YOLOv8 scratches 置信度扫描")
    print(f"模型：{MODEL_PATH}")
    print(f"图片数：{len(images)}")
    print(f"真实标签目标数：{sum(len(x) for x in gt_by_image.values())}")
    print(f"推理 conf={INFERENCE_CONF}, NMS IoU={NMS_IOU}, imgsz={IMGSZ}")
    print("提醒：历史验证记录曾出现 121 个 scratches 实例；本次按当前标签实际统计。")
    print("=" * 76)

    try:
        model = YOLO(MODEL_PATH)
        results = model.predict(
            source=str(IMAGE_DIR),
            imgsz=IMGSZ,
            conf=INFERENCE_CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            save=False,
            verbose=False,
        )
    except Exception as e:
        print(f"[模型推理失败] {e}")
        return 2

    raw_by_image: Dict[str, List[Dict]] = {p.name: [] for p in images}
    for result in results:
        image_name = Path(result.path).name
        if image_name not in raw_by_image or result.boxes is None:
            continue
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for idx, (coords, conf, cls) in enumerate(zip(xyxy, confs, classes)):
            raw_by_image[image_name].append({
                "pred_index": idx,
                "class_id": int(cls),
                "class_name": str(model.names[int(cls)]),
                "confidence": float(conf),
                "xyxy": tuple(float(x) for x in coords),
                "x1": float(coords[0]), "y1": float(coords[1]),
                "x2": float(coords[2]), "y2": float(coords[3]),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    for image_name, dets in raw_by_image.items():
        for det in dets:
            candidate_rows.append({
                "image": image_name,
                "pred_index": det["pred_index"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": f'{det["confidence"]:.6f}',
                "x1": f'{det["x1"]:.2f}', "y1": f'{det["y1"]:.2f}',
                "x2": f'{det["x2"]:.2f}', "y2": f'{det["y2"]:.2f}',
            })
    write_csv(
        OUTPUT_DIR / "candidate_predictions_conf005.csv",
        candidate_rows,
        ["image", "pred_index", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
    )

    rows = []
    for conf_threshold in CONF_THRESHOLDS:
        for scheme in ("baseline", "filter_0.25", "filter_0.20"):
            current_by_image = {}
            for image_name, all_dets in raw_by_image.items():
                selected = [d for d in all_dets if d["confidence"] >= conf_threshold]
                if scheme == "baseline":
                    chosen = selected
                elif scheme == "filter_0.25":
                    chosen = custom_filter(selected, 0.25)
                else:
                    chosen = custom_filter(selected, 0.20)
                current_by_image[image_name] = chosen

            for match_iou in MATCH_IOUS:
                metrics = evaluate(current_by_image, gt_by_image, match_iou)
                rows.append({
                    "scheme": scheme,
                    "confidence_threshold": f"{conf_threshold:.2f}",
                    "match_iou_threshold": f"{match_iou:.2f}",
                    **metrics,
                    "inference_conf": f"{INFERENCE_CONF:.2f}",
                    "nms_iou": f"{NMS_IOU:.2f}",
                })

    fields = [
        "scheme", "confidence_threshold", "match_iou_threshold",
        "images", "ground_truth_count", "prediction_count",
        "TP", "FP", "FN", "precision", "recall", "F1",
        "inference_conf", "nms_iou",
    ]
    write_csv(OUTPUT_DIR / "confidence_sweep_metrics.csv", rows, fields)

    # 每个 scheme 在每个匹配 IoU 下按 F1 排序，输出最优阈值（仅限 0.05–0.70）。
    best_rows = []
    for scheme in ("baseline", "filter_0.25", "filter_0.20"):
        for match_iou in MATCH_IOUS:
            subset = [
                r for r in rows
                if r["scheme"] == scheme
                and r["match_iou_threshold"] == f"{match_iou:.2f}"
            ]
            best = max(subset, key=lambda r: (r["F1"], r["precision"], r["recall"]))
            best_rows.append(best)
    write_csv(OUTPUT_DIR / "best_thresholds_by_f1.csv", best_rows, fields)

    notes = [
        "本次置信度扫描使用当前 best.pt，推理 conf=0.05，NMS IoU=0.80，imgsz=640。",
        "置信度扫描范围为 0.05 到 0.70，步长 0.05。",
        "baseline 为模型内置 NMS 输出；filter_0.25 和 filter_0.20 分别使用对应自定义后处理参数。",
        "match_iou_threshold 为预测框与真实标签的匹配标准，不是 NMS IoU。",
        "输出指标是固定阈值下的 Precision/Recall/F1，不等同于 Ultralytics 的 mAP。",
        "如果真实标签数与预期不符，请先确认标签目录对应当前 validation 图片集。",
    ]
    (OUTPUT_DIR / "README.txt").write_text("\n".join(notes), encoding="utf-8")

    print("\n扫描完成。按 IoU=0.50 的 F1 最优配置：")
    print(f"{'方案':16s} {'conf':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'P':>8s} {'R':>8s} {'F1':>8s}")
    for row in best_rows:
        if row["match_iou_threshold"] == "0.50":
            print(
                f"{row['scheme']:16s} {row['confidence_threshold']:>6s} "
                f"{row['TP']:5d} {row['FP']:5d} {row['FN']:5d} "
                f"{row['precision']:8.4f} {row['recall']:8.4f} {row['F1']:8.4f}"
            )
    print(f"\n详细结果：{OUTPUT_DIR / 'confidence_sweep_metrics.csv'}")
    print(f"最优阈值：{OUTPUT_DIR / 'best_thresholds_by_f1.csv'}")
    print(f"候选框清单：{OUTPUT_DIR / 'candidate_predictions_conf005.csv'}")
    print("把 best_thresholds_by_f1.csv 和 confidence_sweep_metrics.csv 发回即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
