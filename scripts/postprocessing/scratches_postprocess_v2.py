# -*- coding: utf-8 -*-
"""
YOLOv8 钢材表面划痕检测：保守的重复框二次筛选脚本

说明：
1. “处理前”是 YOLOv8 内置 NMS 之后、本文自定义筛选之前的结果。
2. 自定义筛选只比较同类别框，并采用偏保守的几何规则。
3. 对细长框会特别检查窄边方向的中心距离，尽量避免误删彼此相邻的平行划痕。
4. 任何仅靠框几何信息的规则都不能保证完全区分重复框与真实相邻目标，请人工检查对比图和 suppressed_boxes.csv。

运行环境：Windows + Python + ultralytics + OpenCV
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


# =========================
# 1. 用户已确认的本地路径
# =========================
MODEL_PATH = r"E:\yolo_project\runs\neu_defect_clean\weights\best.pt"
SOURCE_DIR = r"D:\yolov8\NEU-DET\validation\images\scratches"
OUTPUT_DIR = r"E:\yolo_project\runs\detect\scratches_postprocess_compare_v2"

# =========================
# 2. 推理参数（可调整）
# =========================
IMGSZ = 640
CONF = 0.30

# 提高内置 NMS IoU 通常会保留更多重叠候选框，之后再由保守的自定义规则筛选。
# 这不是已验证的最佳值；若发现内置 NMS 仍提前压掉重叠候选，可尝试 0.85。
NMS_IOU = 0.80
MAX_DET = 1000

# =========================
# 3. 自定义重复框规则（可调整）
# =========================
# 细长框的长边/短边比例达到该值时，启用细长目标保护逻辑。
ELONGATED_ASPECT_RATIO = 2.2

# 对水平细长框，比较 y 方向中心距离；对垂直细长框，比较 x 方向中心距离。
# 只有窄边方向中心非常接近时，才考虑把两个细长框判为重复框。
# 数值越小越保守，越能保护相邻平行划痕，但也可能漏掉一些重复框。
ELONGATED_PERP_CENTER_MAX = 0.20

# 两个细长框在长边方向上的重叠长度 / 较短框长边长度。
ELONGATED_LONG_OVERLAP_MIN = 0.80
ELONGATED_LONG_CENTER_MAX = 0.60
ELONGATED_AREA_RATIO_MIN = 0.35
ELONGATED_LONG_SIZE_RATIO_MIN = 0.40
ELONGATED_THICKNESS_RATIO_MIN = 0.50

# 普通框：需要很高的 IoU、较近的中心点和相似的尺寸才判为重复。
GENERAL_IOU_MIN = 0.82
GENERAL_CENTER_DIST_MAX = 0.25
GENERAL_WIDTH_RATIO_MIN = 0.55
GENERAL_HEIGHT_RATIO_MIN = 0.55

# 近乎完全包含的框：还要同时满足中心距离近、面积比例不能过小。
CONTAINMENT_MIN = 0.94
CONTAINMENT_CENTER_DIST_MAX = 0.18
CONTAINMENT_AREA_RATIO_MIN = 0.60
CONTAINMENT_WIDTH_RATIO_MIN = 0.55
CONTAINMENT_HEIGHT_RATIO_MIN = 0.55


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dependencies_and_paths() -> None:
    model_path = Path(MODEL_PATH)
    source_path = Path(SOURCE_DIR)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到模型权重：\n{MODEL_PATH}\n"
            "请确认 best.pt 的实际位置与脚本中的 MODEL_PATH 一致。"
        )
    if not source_path.is_dir():
        raise FileNotFoundError(
            f"找不到图片目录：\n{SOURCE_DIR}\n"
            "请确认验证集图片目录与脚本中的 SOURCE_DIR 一致。"
        )

    images = [
        p for p in source_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(
            f"目录中没有找到支持的图片文件：\n{SOURCE_DIR}"
        )


def safe_float(value) -> float:
    return float(value)


def box_geometry(a: Dict, b: Dict) -> Dict[str, float]:
    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]

    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    area_a = aw * ah
    area_b = bw * bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    smaller_area = min(area_a, area_b)
    ios = inter / smaller_area if smaller_area > 0 else 0.0

    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    center_dist = math.hypot(acx - bcx, acy - bcy)

    diag_a = math.hypot(aw, ah)
    diag_b = math.hypot(bw, bh)
    min_diag = max(1.0, min(diag_a, diag_b))

    return {
        "aw": aw,
        "ah": ah,
        "bw": bw,
        "bh": bh,
        "area_a": area_a,
        "area_b": area_b,
        "iou": iou,
        "ios": ios,
        "acx": acx,
        "acy": acy,
        "bcx": bcx,
        "bcy": bcy,
        "center_dist": center_dist,
        "min_diag": min_diag,
        "width_ratio": min(aw, bw) / max(aw, bw),
        "height_ratio": min(ah, bh) / max(ah, bh),
        "area_ratio": min(area_a, area_b) / max(area_a, area_b),
        "inter_w": inter_w,
        "inter_h": inter_h,
    }


def duplicate_reason(a: Dict, b: Dict) -> Optional[str]:
    """
    若认为 b 与 a 是同一个目标的重复定位，返回规则名称；否则返回 None。
    调用前必须保证 a、b 的类别相同。
    """
    g = box_geometry(a, b)

    aw, ah = g["aw"], g["ah"]
    bw, bh = g["bw"], g["bh"]

    a_horizontal = (aw / ah) >= ELONGATED_ASPECT_RATIO
    b_horizontal = (bw / bh) >= ELONGATED_ASPECT_RATIO
    a_vertical = (ah / aw) >= ELONGATED_ASPECT_RATIO
    b_vertical = (bh / bw) >= ELONGATED_ASPECT_RATIO

    # 细长且同为水平框：优先检查窄边方向（y 方向）的中心距离。
    # 若中心线在窄边方向有明显分离，则保留，避免误删相邻平行划痕。
    if a_horizontal and b_horizontal:
        perp_center = abs(g["acy"] - g["bcy"])
        min_thickness = max(1.0, min(ah, bh))
        long_center = abs(g["acx"] - g["bcx"])
        min_long = max(1.0, min(aw, bw))
        long_overlap_ratio = g["inter_w"] / min_long

        if perp_center > ELONGATED_PERP_CENTER_MAX * min_thickness:
            return None

        if (
            long_overlap_ratio >= ELONGATED_LONG_OVERLAP_MIN
            and long_center <= ELONGATED_LONG_CENTER_MAX * min_long
            and g["area_ratio"] >= ELONGATED_AREA_RATIO_MIN
            and g["width_ratio"] >= ELONGATED_LONG_SIZE_RATIO_MIN
            and g["height_ratio"] >= ELONGATED_THICKNESS_RATIO_MIN
        ):
            return "elongated_horizontal_duplicate"

        # 同方向细长框不满足保守条件时，不再用通用 IoU 规则强行删除。
        return None

    # 细长且同为垂直框：优先检查窄边方向（x 方向）的中心距离。
    if a_vertical and b_vertical:
        perp_center = abs(g["acx"] - g["bcx"])
        min_thickness = max(1.0, min(aw, bw))
        long_center = abs(g["acy"] - g["bcy"])
        min_long = max(1.0, min(ah, bh))
        long_overlap_ratio = g["inter_h"] / min_long

        if perp_center > ELONGATED_PERP_CENTER_MAX * min_thickness:
            return None

        if (
            long_overlap_ratio >= ELONGATED_LONG_OVERLAP_MIN
            and long_center <= ELONGATED_LONG_CENTER_MAX * min_long
            and g["area_ratio"] >= ELONGATED_AREA_RATIO_MIN
            and g["height_ratio"] >= ELONGATED_LONG_SIZE_RATIO_MIN
            and g["width_ratio"] >= ELONGATED_THICKNESS_RATIO_MIN
        ):
            return "elongated_vertical_duplicate"

        return None

    # 非同方向细长框或较普通的框：采用更严格的高 IoU 规则。
    if (
        g["iou"] >= GENERAL_IOU_MIN
        and g["center_dist"] <= GENERAL_CENTER_DIST_MAX * g["min_diag"]
        and g["width_ratio"] >= GENERAL_WIDTH_RATIO_MIN
        and g["height_ratio"] >= GENERAL_HEIGHT_RATIO_MIN
    ):
        return "high_iou_similar_size"

    # 处理一个框几乎完全包含另一个框的情况，但必须有相近中心与相似尺寸。
    if (
        g["ios"] >= CONTAINMENT_MIN
        and g["center_dist"] <= CONTAINMENT_CENTER_DIST_MAX * g["min_diag"]
        and g["area_ratio"] >= CONTAINMENT_AREA_RATIO_MIN
        and g["width_ratio"] >= CONTAINMENT_WIDTH_RATIO_MIN
        and g["height_ratio"] >= CONTAINMENT_HEIGHT_RATIO_MIN
    ):
        return "near_complete_containment"

    return None


def apply_conservative_filter(detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    按置信度从高到低遍历。
    仅在同类别框之间执行比较；被抑制项会记录其保留框及规则。
    """
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict] = []
    suppressed: List[Dict] = []

    for candidate in ordered:
        duplicate_of = None
        reason = None

        for keeper in kept:
            if candidate["class_id"] != keeper["class_id"]:
                continue
            reason = duplicate_reason(keeper, candidate)
            if reason is not None:
                duplicate_of = keeper
                break

        if duplicate_of is None:
            kept.append(candidate)
        else:
            record = dict(candidate)
            record["suppression_reason"] = reason
            record["keeper"] = duplicate_of
            suppressed.append(record)

    # 保持输出次序更接近原始检测顺序，便于对照。
    kept.sort(key=lambda d: d["index"])
    suppressed.sort(key=lambda d: d["index"])
    return kept, suppressed


def get_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def extract_detections(result, names) -> List[Dict]:
    detections: List[Dict] = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for i, (coords, confidence, class_id) in enumerate(
        zip(xyxy, confidences, classes)
    ):
        detections.append({
            "index": int(i),
            "class_id": int(class_id),
            "class_name": get_class_name(names, int(class_id)),
            "confidence": safe_float(confidence),
            "xyxy": tuple(float(v) for v in coords),
        })

    return detections


def draw_detections(image, detections: List[Dict], color: Tuple[int, int, int]):
    canvas = image.copy()
    height, width = canvas.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_x = max(0, min(x1, width - text_w - 4))
        label_y = y1 - 5
        if label_y - text_h - baseline < 0:
            label_y = min(height - baseline - 2, y1 + text_h + 5)

        bg_top = max(0, label_y - text_h - baseline - 2)
        bg_bottom = min(height - 1, label_y + baseline + 2)
        bg_right = min(width - 1, label_x + text_w + 4)
        cv2.rectangle(
            canvas,
            (label_x, bg_top),
            (bg_right, bg_bottom),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (label_x + 2, label_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return canvas


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def detection_csv_row(image_name: str, det: Dict) -> Dict:
    x1, y1, x2, y2 = det["xyxy"]
    return {
        "image": image_name,
        "detection_index": det["index"],
        "class_id": det["class_id"],
        "class_name": det["class_name"],
        "confidence": f'{det["confidence"]:.6f}',
        "x1": f"{x1:.2f}",
        "y1": f"{y1:.2f}",
        "x2": f"{x2:.2f}",
        "y2": f"{y2:.2f}",
    }


def suppressed_csv_row(image_name: str, det: Dict) -> Dict:
    x1, y1, x2, y2 = det["xyxy"]
    keeper = det["keeper"]
    kx1, ky1, kx2, ky2 = keeper["xyxy"]
    return {
        "image": image_name,
        "suppressed_index": det["index"],
        "class_id": det["class_id"],
        "class_name": det["class_name"],
        "confidence": f'{det["confidence"]:.6f}',
        "x1": f"{x1:.2f}",
        "y1": f"{y1:.2f}",
        "x2": f"{x2:.2f}",
        "y2": f"{y2:.2f}",
        "suppression_reason": det["suppression_reason"],
        "keeper_index": keeper["index"],
        "keeper_confidence": f'{keeper["confidence"]:.6f}',
        "keeper_x1": f"{kx1:.2f}",
        "keeper_y1": f"{ky1:.2f}",
        "keeper_x2": f"{kx2:.2f}",
        "keeper_y2": f"{ky2:.2f}",
    }


def main() -> int:
    try:
        ensure_dependencies_and_paths()
    except Exception as exc:
        print(f"[错误] {exc}")
        return 1

    output_root = Path(OUTPUT_DIR)
    before_dir = output_root / "01_before_custom_filter"
    after_dir = output_root / "02_after_custom_filter"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("YOLOv8 划痕检测：保守的自定义重复框筛选")
    print(f"模型：{MODEL_PATH}")
    print(f"图片：{SOURCE_DIR}")
    print(f"输出：{OUTPUT_DIR}")
    print(
        f"参数：imgsz={IMGSZ}, conf={CONF}, 内置 NMS iou={NMS_IOU}, "
        f"max_det={MAX_DET}"
    )
    print("说明：处理前结果已经过 YOLOv8 内置 NMS。")
    print("=" * 72)

    try:
        model = YOLO(MODEL_PATH)
        results = model.predict(
            source=SOURCE_DIR,
            imgsz=IMGSZ,
            conf=CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            save=False,
            verbose=False,
        )
    except Exception as exc:
        print(f"[错误] 模型加载或推理失败：{exc}")
        return 2

    summary_rows: List[Dict] = []
    before_rows: List[Dict] = []
    after_rows: List[Dict] = []
    suppressed_rows: List[Dict] = []

    total_before = 0
    total_after = 0
    total_suppressed = 0
    total_before_scratches = 0
    total_after_scratches = 0
    total_suppressed_scratches = 0

    for image_number, result in enumerate(results, start=1):
        image_path = Path(result.path)
        image_name = image_path.name
        original = result.orig_img

        before = extract_detections(result, model.names)
        after, suppressed = apply_conservative_filter(before)

        # 处理前（模型内置 NMS 之后、自定义筛选之前）：橙蓝色框。
        before_image = draw_detections(original, before, (255, 140, 0))
        before_out = before_dir / image_name
        if not cv2.imwrite(str(before_out), before_image):
            print(f"[警告] 无法保存处理前图片：{before_out}")

        # 处理后：绿色框。
        after_image = draw_detections(original, after, (0, 180, 0))
        after_out = after_dir / image_name
        if not cv2.imwrite(str(after_out), after_image):
            print(f"[警告] 无法保存处理后图片：{after_out}")

        for det in before:
            before_rows.append(detection_csv_row(image_name, det))
        for det in after:
            after_rows.append(detection_csv_row(image_name, det))
        for det in suppressed:
            suppressed_rows.append(suppressed_csv_row(image_name, det))

        before_scratches = sum(
            1 for d in before if d["class_name"] == "scratches"
        )
        after_scratches = sum(
            1 for d in after if d["class_name"] == "scratches"
        )
        suppressed_scratches = sum(
            1 for d in suppressed if d["class_name"] == "scratches"
        )

        summary_rows.append({
            "image": image_name,
            "before_total_boxes": len(before),
            "after_total_boxes": len(after),
            "suppressed_total_boxes": len(suppressed),
            "before_scratches_boxes": before_scratches,
            "after_scratches_boxes": after_scratches,
            "suppressed_scratches_boxes": suppressed_scratches,
        })

        total_before += len(before)
        total_after += len(after)
        total_suppressed += len(suppressed)
        total_before_scratches += before_scratches
        total_after_scratches += after_scratches
        total_suppressed_scratches += suppressed_scratches

        print(
            f"[{image_number:03d}/{len(results):03d}] {image_name} | "
            f"处理前 {len(before):2d} 框 -> 处理后 {len(after):2d} 框 | "
            f"筛除 {len(suppressed):2d} 框"
        )

    summary_fields = [
        "image",
        "before_total_boxes",
        "after_total_boxes",
        "suppressed_total_boxes",
        "before_scratches_boxes",
        "after_scratches_boxes",
        "suppressed_scratches_boxes",
    ]
    detection_fields = [
        "image",
        "detection_index",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    suppressed_fields = [
        "image",
        "suppressed_index",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "suppression_reason",
        "keeper_index",
        "keeper_confidence",
        "keeper_x1",
        "keeper_y1",
        "keeper_x2",
        "keeper_y2",
    ]

    write_csv(output_root / "summary.csv", summary_fields, summary_rows)
    write_csv(output_root / "detections_before.csv", detection_fields, before_rows)
    write_csv(output_root / "detections_after.csv", detection_fields, after_rows)
    write_csv(
        output_root / "suppressed_boxes.csv",
        suppressed_fields,
        suppressed_rows,
    )

    print("\n" + "=" * 72)
    print("运行完成")
    print(f"处理图片数：{len(results)}")
    print(f"所有类别：处理前 {total_before} 框，处理后 {total_after} 框，筛除 {total_suppressed} 框")
    print(
        "scratches 类："
        f"处理前 {total_before_scratches} 框，"
        f"处理后 {total_after_scratches} 框，"
        f"筛除 {total_suppressed_scratches} 框"
    )
    print(f"处理前图片：{before_dir}")
    print(f"处理后图片：{after_dir}")
    print(f"逐图统计：{output_root / 'summary.csv'}")
    print(f"筛除明细：{output_root / 'suppressed_boxes.csv'}")
    print(f"处理前框清单：{output_root / 'detections_before.csv'}")
    print(f"处理后框清单：{output_root / 'detections_after.csv'}")
    print("=" * 72)
    print("请重点人工复核相邻平行划痕，以及 scratches_290.jpg 中低置信度真实目标是否保留。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
