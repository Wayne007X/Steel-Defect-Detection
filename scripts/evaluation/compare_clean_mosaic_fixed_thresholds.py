# -*- coding: utf-8 -*-
"""
NEU-DET：clean 与 mosaic000 固定阈值公平对照（仅推理，不训练）

比较条件：
- imgsz = 640
- 推理 NMS IoU = 0.30
- 评估匹配 IoU = 0.50
- conf = 0.20 / 0.25
- max_det = 300
- 验证集应为 360 张图片、854 个真实目标

运行环境：Windows + 已安装 ultralytics、torch、Pillow、numpy。
运行前请确认脚本顶部路径与本机一致。
"""

from __future__ import annotations

import csv
import gc
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from ultralytics import YOLO

# ===================== 路径配置：按当前项目报告填写 =====================
CLEAN_WEIGHT = Path(r"E:\yolo_project\runs\neu_defect_clean\weights\best.pt")
MOSAIC_WEIGHT = Path(r"E:\yolo_project\runs\neu_defect_mosaic000\weights\best.pt")

VAL_IMAGE_ROOT = Path(r"D:\yolov8\NEU-DET\validation\images")
VAL_LABEL_ROOT = Path(r"D:\yolov8\NEU-DET\validation\labels")

OUTPUT_DIR = Path(r"E:\yolo_project\runs\clean_mosaic_fixed_threshold_compare_20261026")

# ===================== 固定评估设置：不要随意改动 =====================
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

IMGSZ = 640
PRED_FLOOR_CONF = 0.001  # 先缓存低阈值候选，再按 0.20 / 0.25 统计
NMS_IOU = 0.30
MATCH_IOU = 0.50
CONF_THRESHOLDS = (0.20, 0.25)
MAX_DET = 300
BATCH = 2
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 0

EXPECTED_IMAGES = 360
EXPECTED_GROUND_TRUTHS = 854


def log(msg: str) -> None:
    print(msg, flush=True)


def image_key(path: Path) -> str:
    """返回相对 validation/images 的规范化路径，保留类别子目录。"""
    path = path.resolve()
    root = VAL_IMAGE_ROOT.resolve()
    try:
        return path.relative_to(root).as_posix().casefold()
    except ValueError as exc:
        raise ValueError(f"预测结果路径不在验证图片目录内：{path}") from exc


def get_images() -> list[Path]:
    images = sorted(
        [
            p for p in VAL_IMAGE_ROOT.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.relative_to(VAL_IMAGE_ROOT).as_posix().casefold(),
    )
    if not images:
        raise FileNotFoundError(f"验证图片目录中没有找到图片：{VAL_IMAGE_ROOT}")
    return images


def load_ground_truths(images: list[Path]) -> dict[str, list[dict]]:
    gt_by_image: dict[str, list[dict]] = {}
    missing_labels = []
    total_gt = 0

    for image_path in images:
        key = image_key(image_path)
        rel = image_path.relative_to(VAL_IMAGE_ROOT)
        label_path = (VAL_LABEL_ROOT / rel).with_suffix(".txt")

        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue

        with Image.open(image_path) as im:
            width, height = im.size

        boxes = []
        for line_no, line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"标签格式错误：{label_path} 第 {line_no} 行：{line}")

            class_id = int(float(parts[0]))
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(
                    f"类别编号越界：{label_path} 第 {line_no} 行，class_id={class_id}"
                )

            xc, yc, bw, bh = map(float, parts[1:5])
            boxes.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "x1": (xc - bw / 2.0) * width,
                "y1": (yc - bh / 2.0) * height,
                "x2": (xc + bw / 2.0) * width,
                "y2": (yc + bh / 2.0) * height,
            })

        gt_by_image[key] = boxes
        total_gt += len(boxes)

    if missing_labels:
        examples = "\n".join(missing_labels[:10])
        raise FileNotFoundError(
            f"有 {len(missing_labels)} 张图片没有找到对应标签文件。示例：\n{examples}"
        )

    log(f"验证图片数：{len(images)}；真实目标数：{total_gt}")
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"验证图片数应为 {EXPECTED_IMAGES}，实际为 {len(images)}。"
            "请先确认数据目录和划分，不要继续比较。"
        )
    if total_gt != EXPECTED_GROUND_TRUTHS:
        raise RuntimeError(
            f"验证集真实目标数应为 {EXPECTED_GROUND_TRUTHS}，实际为 {total_gt}。"
            "请先核对标签路径、标签内容和数据划分，不要继续比较。"
        )
    return gt_by_image


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


def predict_model(model_name: str, weight_path: Path, images: list[Path]) -> dict[str, list[dict]]:
    if not weight_path.exists():
        raise FileNotFoundError(f"模型权重不存在：{weight_path}")

    log(f"\n[{model_name}] 加载权重：{weight_path}")
    model = YOLO(str(weight_path))
    predictions: dict[str, list[dict]] = {}

    results = model.predict(
        source=[str(p) for p in images],
        stream=True,
        imgsz=IMGSZ,
        batch=BATCH,
        conf=PRED_FLOOR_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        device=DEVICE,
        workers=WORKERS,
        save=False,
        verbose=False,
        augment=False,
    )

    expected = {image_key(p) for p in images}
    for index, result in enumerate(results, start=1):
        result_path = Path(result.path)
        if not result_path.is_absolute():
            result_path = (VAL_IMAGE_ROOT / result_path).resolve()
        key = image_key(result_path)

        boxes = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)

            for coords, conf, class_id in zip(xyxy, confs, classes):
                if 0 <= int(class_id) < len(CLASS_NAMES):
                    boxes.append({
                        "class_id": int(class_id),
                        "class_name": CLASS_NAMES[int(class_id)],
                        "confidence": float(conf),
                        "x1": float(coords[0]),
                        "y1": float(coords[1]),
                        "x2": float(coords[2]),
                        "y2": float(coords[3]),
                    })

        predictions[key] = boxes
        if index % 60 == 0 or index == len(images):
            log(f"[{model_name}] 推理进度：{index}/{len(images)}")

    missing = sorted(expected - set(predictions))
    extra = sorted(set(predictions) - expected)
    if missing or extra:
        raise RuntimeError(
            f"{model_name} 推理结果与验证图片不匹配："
            f"缺少 {len(missing)} 张，额外 {len(extra)} 张。"
            f"{'缺失示例：' + missing[0] if missing else ''}"
        )

    del results, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def metrics(tp: int, fp: int, fn: int) -> dict:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def evaluate(
    model_name: str,
    predictions: dict[str, list[dict]],
    gt_by_image: dict[str, list[dict]],
    conf_threshold: float,
) -> tuple[dict, list[dict]]:
    counts = {
        i: {"GT": 0, "TP": 0, "FP": 0, "FN": 0}
        for i in range(len(CLASS_NAMES))
    }

    for image_key_value, gts in gt_by_image.items():
        for gt in gts:
            counts[gt["class_id"]]["GT"] += 1

        preds = [
            p for p in predictions.get(image_key_value, [])
            if p["confidence"] >= conf_threshold
        ]
        preds.sort(key=lambda p: p["confidence"], reverse=True)
        unmatched_gt = set(range(len(gts)))

        for pred in preds:
            best_iou = 0.0
            best_gt_idx = None
            for gt_idx in unmatched_gt:
                gt = gts[gt_idx]
                if gt["class_id"] != pred["class_id"]:
                    continue
                overlap = box_iou(pred, gt)
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt_idx = gt_idx

            if best_gt_idx is not None and best_iou >= MATCH_IOU:
                counts[pred["class_id"]]["TP"] += 1
                unmatched_gt.remove(best_gt_idx)
            else:
                counts[pred["class_id"]]["FP"] += 1

        for gt_idx in unmatched_gt:
            counts[gts[gt_idx]["class_id"]]["FN"] += 1

    per_class_rows = []
    f1_values = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        row = {
            "model": model_name,
            "confidence_threshold": conf_threshold,
            "nms_iou": NMS_IOU,
            "match_iou": MATCH_IOU,
            "class_id": class_id,
            "class_name": class_name,
            **counts[class_id],
        }
        row.update(metrics(row["TP"], row["FP"], row["FN"]))
        per_class_rows.append(row)
        f1_values.append(row["F1"])

    tp = sum(v["TP"] for v in counts.values())
    fp = sum(v["FP"] for v in counts.values())
    fn = sum(v["FN"] for v in counts.values())
    gt_total = sum(v["GT"] for v in counts.values())
    overall = {
        "model": model_name,
        "confidence_threshold": conf_threshold,
        "imgsz": IMGSZ,
        "nms_iou": NMS_IOU,
        "match_iou": MATCH_IOU,
        "images": len(gt_by_image),
        "GT": gt_total,
        **metrics(tp, fp, fn),
        "macro_F1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
    }
    return overall, per_class_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    required = [CLEAN_WEIGHT, MOSAIC_WEIGHT, VAL_IMAGE_ROOT, VAL_LABEL_ROOT]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("以下路径不存在，请先修改脚本顶部配置：\n" + "\n".join(missing))

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"输出目录已存在，为避免覆盖旧结果，脚本停止：{OUTPUT_DIR}\n"
            "如需重跑，请修改 OUTPUT_DIR 为新的目录名。"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    images = get_images()
    gt_by_image = load_ground_truths(images)

    overall_rows = []
    class_rows = []

    for model_name, weight_path in [
        ("neu_defect_clean", CLEAN_WEIGHT),
        ("neu_defect_mosaic000", MOSAIC_WEIGHT),
    ]:
        preds = predict_model(model_name, weight_path, images)
        for conf in CONF_THRESHOLDS:
            overall, per_class = evaluate(model_name, preds, gt_by_image, conf)
            overall_rows.append(overall)
            class_rows.extend(per_class)

    overall_csv = OUTPUT_DIR / "fixed_threshold_overall.csv"
    class_csv = OUTPUT_DIR / "fixed_threshold_per_class.csv"
    write_csv(overall_csv, overall_rows)
    write_csv(class_csv, class_rows)

    # 生成便于人工查看的摘要
    by_key = {(r["model"], r["confidence_threshold"]): r for r in overall_rows}
    summary_lines = [
        "NEU-DET clean vs mosaic000 固定阈值对照",
        f"验证图片：{len(images)}；真实目标：{sum(len(v) for v in gt_by_image.values())}",
        f"imgsz={IMGSZ}, NMS IoU={NMS_IOU}, 匹配 IoU={MATCH_IOU}, max_det={MAX_DET}",
        "",
        "差值定义：mosaic000 - clean",
    ]
    for conf in CONF_THRESHOLDS:
        clean = by_key[("neu_defect_clean", conf)]
        mosaic = by_key[("neu_defect_mosaic000", conf)]
        summary_lines.extend([
            f"conf={conf:.2f}",
            f"  clean:    P={clean['precision']:.4f}, R={clean['recall']:.4f}, "
            f"F1={clean['F1']:.4f}, TP={clean['TP']}, FP={clean['FP']}, FN={clean['FN']}, "
            f"Macro-F1={clean['macro_F1']:.4f}",
            f"  mosaic000: P={mosaic['precision']:.4f}, R={mosaic['recall']:.4f}, "
            f"F1={mosaic['F1']:.4f}, TP={mosaic['TP']}, FP={mosaic['FP']}, FN={mosaic['FN']}, "
            f"Macro-F1={mosaic['macro_F1']:.4f}",
            f"  delta:    P={mosaic['precision']-clean['precision']:+.4f}, "
            f"R={mosaic['recall']-clean['recall']:+.4f}, "
            f"F1={mosaic['F1']-clean['F1']:+.4f}, "
            f"TP={mosaic['TP']-clean['TP']:+d}, FP={mosaic['FP']-clean['FP']:+d}, "
            f"FN={mosaic['FN']-clean['FN']:+d}, "
            f"Macro-F1={mosaic['macro_F1']-clean['macro_F1']:+.4f}",
            "",
        ])

    summary_path = OUTPUT_DIR / "comparison_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    log("\n对照完成。输出文件：")
    log(f"  {overall_csv}")
    log(f"  {class_csv}")
    log(f"  {summary_path}")
    log("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
