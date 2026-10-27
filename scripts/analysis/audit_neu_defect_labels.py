# -*- coding: utf-8 -*-
"""
NEU-DET 六类缺陷标签自动审计脚本
不使用 Cursor Agent，不运行模型，不修改数据。

输出：
E:\\yolo_project\\runs\\detect\\label_audit\\label_audit_class_summary.csv
E:\\yolo_project\\runs\\detect\\label_audit\\label_audit_image_summary.csv
E:\\yolo_project\\runs\\detect\\label_audit\\label_audit_box_details.csv

检查：
- train/val 各类别实例数、含该类别的图片数
- 缺失标签、无标注图片、无对应图片的标签文件
- YOLO 归一化框宽高、面积、长宽比
- 细小目标比例，帮助判断 crazing 漏检是否可能与目标尺度有关
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

DATASET_ROOT = Path(r"D:\yolov8\NEU-DET")
OUTPUT_DIR = Path(r"E:\yolo_project\runs\detect\label_audit")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_files(root: Path, suffixes: set[str]) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"找不到目录：{root}")
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )


def build_stem_map(paths: List[Path], label: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    duplicates = defaultdict(list)
    for p in paths:
        if p.stem in mapping:
            duplicates[p.stem].extend([mapping[p.stem], p])
        else:
            mapping[p.stem] = p
    if duplicates:
        examples = []
        for stem, ps in list(duplicates.items())[:10]:
            examples.append(f"{stem}: " + " | ".join(str(p) for p in ps))
        raise RuntimeError(
            f"{label}中发现同名文件，无法仅凭文件名安全匹配：\n" + "\n".join(examples)
        )
    return mapping


def read_yolo_label(label_path: Path) -> List[Dict]:
    boxes = []
    with label_path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"标签格式错误：{label_path} 第 {line_no} 行：{line}")
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            if not 0 <= cls < len(CLASS_NAMES):
                raise ValueError(
                    f"类别编号超出范围：{label_path} 第 {line_no} 行 class_id={cls}"
                )
            if bw < 0 or bh < 0:
                raise ValueError(f"标签宽高为负数：{label_path} 第 {line_no} 行")
            boxes.append({
                "class_id": cls,
                "x_center": xc,
                "y_center": yc,
                "width": bw,
                "height": bh,
                "area": bw * bh,
                "aspect_ratio": bw / bh if bh > 0 else "",
                "line_no": line_no,
            })
    return boxes


def summarize(values: List[float], prefix: str) -> Dict:
    if not values:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_median": "",
            f"{prefix}_min": "",
            f"{prefix}_max": "",
        }
    return {
        f"{prefix}_mean": round(mean(values), 6),
        f"{prefix}_median": round(median(values), 6),
        f"{prefix}_min": round(min(values), 6),
        f"{prefix}_max": round(max(values), 6),
    }


def audit_split(split: str):
    image_root = DATASET_ROOT / split / "images"
    label_root = DATASET_ROOT / split / "labels"

    images = find_files(image_root, IMAGE_EXTS)
    labels = find_files(label_root, {".txt"})
    image_map = build_stem_map(images, f"{split} 图片")
    label_map = build_stem_map(labels, f"{split} 标签")

    missing_labels = sorted(set(image_map) - set(label_map))
    orphan_labels = sorted(set(label_map) - set(image_map))

    class_instances = [0] * len(CLASS_NAMES)
    class_images = [0] * len(CLASS_NAMES)
    class_boxes = [[] for _ in CLASS_NAMES]
    image_rows = []
    box_rows = []
    empty_label_images = 0
    bad_bounds = 0

    for stem, image_path in image_map.items():
        label_path = label_map.get(stem)
        if label_path is None:
            image_rows.append({
                "split": split,
                "image": image_path.name,
                "label_file": "",
                "has_label_file": False,
                "instances": "",
                "classes_present": "",
                "status": "missing_label_file",
            })
            continue

        boxes = read_yolo_label(label_path)
        if not boxes:
            empty_label_images += 1

        present = set()
        for box_idx, box in enumerate(boxes):
            cls = box["class_id"]
            present.add(cls)
            class_instances[cls] += 1
            class_boxes[cls].append(box)
            if not (
                0 <= box["x_center"] <= 1
                and 0 <= box["y_center"] <= 1
                and 0 <= box["width"] <= 1
                and 0 <= box["height"] <= 1
            ):
                bad_bounds += 1

            box_rows.append({
                "split": split,
                "image": image_path.name,
                "label_file": str(label_path),
                "box_index": box_idx,
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "x_center_norm": box["x_center"],
                "y_center_norm": box["y_center"],
                "width_norm": box["width"],
                "height_norm": box["height"],
                "area_norm": box["area"],
                "aspect_ratio_w_over_h": box["aspect_ratio"],
                "area_lt_0.01": box["area"] < 0.01,
                "area_lt_0.02": box["area"] < 0.02,
                "area_lt_0.05": box["area"] < 0.05,
                "width_lt_0.05": box["width"] < 0.05,
                "height_lt_0.05": box["height"] < 0.05,
            })

        for cls in present:
            class_images[cls] += 1

        image_rows.append({
            "split": split,
            "image": image_path.name,
            "label_file": str(label_path),
            "has_label_file": True,
            "instances": len(boxes),
            "classes_present": ";".join(CLASS_NAMES[i] for i in sorted(present)),
            "status": "ok" if boxes else "empty_label_file",
        })

    class_rows = []
    for cls, name in enumerate(CLASS_NAMES):
        boxes = class_boxes[cls]
        widths = [b["width"] for b in boxes]
        heights = [b["height"] for b in boxes]
        areas = [b["area"] for b in boxes]
        aspects = [b["aspect_ratio"] for b in boxes if b["aspect_ratio"] != ""]
        count = len(boxes)

        row = {
            "split": split,
            "class_id": cls,
            "class_name": name,
            "images_total": len(images),
            "label_files_total": len(labels),
            "images_with_class": class_images[cls],
            "instances": count,
            "instances_per_image_with_class": round(count / class_images[cls], 4) if class_images[cls] else 0,
            "area_lt_0.01_count": sum(a < 0.01 for a in areas),
            "area_lt_0.01_pct": round(sum(a < 0.01 for a in areas) / count, 4) if count else 0,
            "area_lt_0.02_count": sum(a < 0.02 for a in areas),
            "area_lt_0.02_pct": round(sum(a < 0.02 for a in areas) / count, 4) if count else 0,
            "area_lt_0.05_count": sum(a < 0.05 for a in areas),
            "area_lt_0.05_pct": round(sum(a < 0.05 for a in areas) / count, 4) if count else 0,
            "width_lt_0.05_count": sum(w < 0.05 for w in widths),
            "height_lt_0.05_count": sum(h < 0.05 for h in heights),
        }
        row.update(summarize(widths, "width_norm"))
        row.update(summarize(heights, "height_norm"))
        row.update(summarize(areas, "area_norm"))
        row.update(summarize(aspects, "aspect_ratio"))
        class_rows.append(row)

    integrity = {
        "split": split,
        "images": len(images),
        "label_files": len(labels),
        "images_missing_label": len(missing_labels),
        "label_files_without_image": len(orphan_labels),
        "empty_label_files": empty_label_images,
        "boxes_outside_normalized_range": bad_bounds,
        "missing_label_examples": ";".join(missing_labels[:20]),
        "orphan_label_examples": ";".join(orphan_labels[:20]),
    }
    return class_rows, image_rows, box_rows, integrity


def main() -> int:
    try:
        if not DATASET_ROOT.is_dir():
            raise FileNotFoundError(f"找不到数据集目录：{DATASET_ROOT}")

        all_class_rows, all_image_rows, all_box_rows, integrity_rows = [], [], [], []
        for split in ("train", "validation"):
            class_rows, image_rows, box_rows, integrity = audit_split(split)
            all_class_rows.extend(class_rows)
            all_image_rows.extend(image_rows)
            all_box_rows.extend(box_rows)
            integrity_rows.append(integrity)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_csv(
            OUTPUT_DIR / "label_audit_class_summary.csv",
            all_class_rows,
            list(all_class_rows[0].keys()),
        )
        write_csv(
            OUTPUT_DIR / "label_audit_image_summary.csv",
            all_image_rows,
            list(all_image_rows[0].keys()) if all_image_rows else ["split", "image", "status"],
        )
        write_csv(
            OUTPUT_DIR / "label_audit_box_details.csv",
            all_box_rows,
            [
                "split", "image", "label_file", "box_index", "class_id", "class_name",
                "x_center_norm", "y_center_norm", "width_norm", "height_norm",
                "area_norm", "aspect_ratio_w_over_h", "area_lt_0.01", "area_lt_0.02",
                "area_lt_0.05", "width_lt_0.05", "height_lt_0.05",
            ],
        )
        write_csv(
            OUTPUT_DIR / "label_audit_integrity.csv",
            integrity_rows,
            list(integrity_rows[0].keys()),
        )

        print("=" * 88)
        print("NEU-DET 标签审计完成（没有运行模型或修改数据）")
        for r in integrity_rows:
            print(
                f"{r['split']:10s}: 图片 {r['images']:4d}，标签文件 {r['label_files']:4d}，"
                f"缺失标签 {r['images_missing_label']:3d}，无对应图片标签 {r['label_files_without_image']:3d}，"
                f"空标签 {r['empty_label_files']:3d}，异常归一化框 {r['boxes_outside_normalized_range']:3d}"
            )
        print("\n类别统计：")
        print(f"{'split':10s} {'class':20s} {'images':>7s} {'instances':>10s} {'area<1%':>9s} {'area<2%':>9s} {'area<5%':>9s}")
        for r in all_class_rows:
            print(
                f"{r['split']:10s} {r['class_name']:20s} {r['images_with_class']:7d} "
                f"{r['instances']:10d} {r['area_lt_0.01_pct']:9.1%} "
                f"{r['area_lt_0.02_pct']:9.1%} {r['area_lt_0.05_pct']:9.1%}"
            )

        print(f"\n输出目录：{OUTPUT_DIR}")
        print("请把 label_audit_class_summary.csv 和 label_audit_integrity.csv 发回。")
        return 0
    except Exception as exc:
        print(f"[审计失败] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
