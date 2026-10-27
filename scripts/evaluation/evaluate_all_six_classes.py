# -*- coding: utf-8 -*-
"""
NEU-DET 六类钢材表面缺陷：统一基线 + 四组推理参数定量对比
无需 Cursor Agent，直接在 Cursor 的 PowerShell 终端运行。

模型：
E:\\yolo_project\\runs\\neu_defect_clean\\weights\\best.pt

数据配置：
D:\\yolov8\\NEU-DET\\data.yaml

验证集包含 6 类：
crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches

实验：
A: conf=0.30, NMS IoU=0.30
B: conf=0.40, NMS IoU=0.30
C: conf=0.30, NMS IoU=0.70
D: conf=0.30, NMS IoU=0.45

输出：
E:\\yolo_project\\runs\\detect\\full_six_class_tuning

说明：
- 标准验证使用低 conf=0.001 计算标准 mAP 和验证曲线。
- 参数对比使用固定 conf/NMS IoU，对预测框和真实标签进行一对一匹配。
- match IoU=0.50/0.75 是预测框与真实框的匹配阈值，不是 NMS 参数。
- 本脚本不会修改、覆盖或重新训练 best.pt。
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO


# =========================
# 已确认路径与类别
# =========================
MODEL_PATH = Path(
    r"E:\yolo_project\runs\neu_defect_mosaic050\weights\best.pt"
)
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
IMAGE_ROOT = Path(r"D:\yolov8\NEU-DET\validation\images")
LABEL_ROOT = Path(r"D:\yolov8\NEU-DET\validation\labels")
OUTPUT_ROOT = Path(
    r"E:\yolo_project\runs\detect\full_six_class_tuning_mosaic050"
)
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMGSZ = 640
MAX_DET = 1000

# 固定置信度/NMS IoU 参数组
EXPERIMENTS = [
    {"name": "A_conf030_iou030", "conf": 0.30, "nms_iou": 0.30},
    {"name": "B_conf040_iou030", "conf": 0.40, "nms_iou": 0.30},
    {"name": "C_conf030_iou070", "conf": 0.30, "nms_iou": 0.70},
    {"name": "D_conf030_iou045", "conf": 0.30, "nms_iou": 0.45},
]

MATCH_IOUS = (0.50, 0.75)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def check_paths() -> None:
    for p, label in [
        (MODEL_PATH, "模型权重"),
        (DATA_YAML, "data.yaml"),
        (IMAGE_ROOT, "验证图片目录"),
        (LABEL_ROOT, "验证标签目录"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"找不到{label}：{p}")


def collect_images() -> List[Path]:
    images = sorted(
        p for p in IMAGE_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"没有找到验证图片：{IMAGE_ROOT}")
    return images


def collect_labels() -> Dict[str, List[Path]]:
    by_stem: Dict[str, List[Path]] = defaultdict(list)
    for p in LABEL_ROOT.rglob("*.txt"):
        by_stem[p.stem].append(p)
    return by_stem


def load_ground_truths(images: List[Path]) -> Tuple[Dict[str, List[Dict]], int]:
    labels_by_stem = collect_labels()
    gt_by_image: Dict[str, List[Dict]] = {}
    missing, duplicate = [], []
    total_gt = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"OpenCV 无法读取图片：{image_path}")
        h, w = image.shape[:2]

        candidates = labels_by_stem.get(image_path.stem, [])
        if not candidates:
            missing.append(str(image_path))
            continue
        if len(candidates) > 1:
            duplicate.append((image_path.name, [str(x) for x in candidates]))
            continue

        label_path = candidates[0]
        boxes: List[Dict] = []
        with label_path.open("r", encoding="utf-8-sig") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(
                        f"标签格式错误：{label_path} 第 {line_no} 行：{line}"
                    )

                class_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    raise ValueError(
                        f"标签类别编号超出范围：{label_path} 第 {line_no} 行，"
                        f"class_id={class_id}，当前应为 0–{len(CLASS_NAMES)-1}"
                    )

                boxes.append({
                    "gt_index": len(boxes),
                    "class_id": class_id,
                    "x1": (xc - bw / 2.0) * w,
                    "y1": (yc - bh / 2.0) * h,
                    "x2": (xc + bw / 2.0) * w,
                    "y2": (yc + bh / 2.0) * h,
                })

        gt_by_image[image_path.name] = boxes
        total_gt += len(boxes)

    if missing:
        raise FileNotFoundError(
            f"有 {len(missing)} 张图片没有找到同名标签文件。示例：\n"
            + "\n".join(missing[:15])
            + f"\n请确认标签位于 {LABEL_ROOT} 下，且与图片文件名一致。"
        )
    if duplicate:
        raise RuntimeError(
            "发现同名图片对应多个标签文件，无法安全判断应使用哪一个：\n"
            + "\n".join(f"{name}: {paths}" for name, paths in duplicate[:15])
        )

    return gt_by_image, total_gt


def box_iou(a: Dict, b: Dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def metric_values(tp: int, fp: int, fn: int) -> Dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def evaluate_config(
    predictions_by_image: Dict[str, List[Dict]],
    gt_by_image: Dict[str, List[Dict]],
    match_iou: float,
    experiment_name: str,
    conf: float,
    nms_iou: float,
) -> Tuple[List[Dict], Dict, List[Dict]]:
    class_counts = {
        i: {"TP": 0, "FP": 0, "FN": 0, "GT": 0, "Pred": 0}
        for i in range(len(CLASS_NAMES))
    }
    per_image_rows: List[Dict] = []
    unmatched_rows: List[Dict] = []

    for image_name, gts in gt_by_image.items():
        preds = sorted(
            predictions_by_image.get(image_name, []),
            key=lambda p: p["confidence"],
            reverse=True,
        )
        for gt in gts:
            class_counts[gt["class_id"]]["GT"] += 1
        for pred in preds:
            class_counts[pred["class_id"]]["Pred"] += 1

        unmatched_gt = set(range(len(gts)))
        tp_image = [0] * len(CLASS_NAMES)
        fp_image = [0] * len(CLASS_NAMES)

        for pred in preds:
            candidates = []
            for gt_idx in unmatched_gt:
                gt = gts[gt_idx]
                if pred["class_id"] != gt["class_id"]:
                    continue
                candidates.append((box_iou(pred, gt), gt_idx))

            best_iou, best_gt_idx = max(candidates, default=(0.0, None), key=lambda x: x[0])

            if best_gt_idx is not None and best_iou >= match_iou:
                unmatched_gt.remove(best_gt_idx)
                class_counts[pred["class_id"]]["TP"] += 1
                tp_image[pred["class_id"]] += 1
            else:
                class_counts[pred["class_id"]]["FP"] += 1
                fp_image[pred["class_id"]] += 1
                unmatched_rows.append({
                    "experiment": experiment_name,
                    "conf": conf,
                    "nms_iou": nms_iou,
                    "match_iou": match_iou,
                    "image": image_name,
                    "type": "unmatched_prediction",
                    "class_id": pred["class_id"],
                    "class_name": pred["class_name"],
                    "confidence": pred["confidence"],
                    "x1": pred["x1"], "y1": pred["y1"],
                    "x2": pred["x2"], "y2": pred["y2"],
                    "best_same_class_iou": best_iou,
                })

        for gt_idx in unmatched_gt:
            gt = gts[gt_idx]
            class_counts[gt["class_id"]]["FN"] += 1
            unmatched_rows.append({
                "experiment": experiment_name,
                "conf": conf,
                "nms_iou": nms_iou,
                "match_iou": match_iou,
                "image": image_name,
                "type": "unmatched_ground_truth",
                "class_id": gt["class_id"],
                "class_name": CLASS_NAMES[gt["class_id"]],
                "confidence": "",
                "x1": gt["x1"], "y1": gt["y1"],
                "x2": gt["x2"], "y2": gt["y2"],
                "best_same_class_iou": "",
            })

        for cls_id, cls_name in enumerate(CLASS_NAMES):
            gt_cls = sum(1 for gt in gts if gt["class_id"] == cls_id)
            pred_cls = sum(1 for pred in preds if pred["class_id"] == cls_id)
            tp = tp_image[cls_id]
            fp = fp_image[cls_id]
            fn = gt_cls - tp
            m = metric_values(tp, fp, fn)
            per_image_rows.append({
                "experiment": experiment_name,
                "conf": conf,
                "nms_iou": nms_iou,
                "match_iou": match_iou,
                "image": image_name,
                "class_id": cls_id,
                "class_name": cls_name,
                "GT": gt_cls,
                "Pred": pred_cls,
                **m,
            })

    class_rows = []
    total_tp = total_fp = total_fn = total_gt = total_pred = 0
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        c = class_counts[cls_id]
        m = metric_values(c["TP"], c["FP"], c["FN"])
        class_rows.append({
            "experiment": experiment_name,
            "conf": conf,
            "nms_iou": nms_iou,
            "match_iou": match_iou,
            "class_id": cls_id,
            "class_name": cls_name,
            "GT": c["GT"],
            "Pred": c["Pred"],
            **m,
        })
        total_tp += c["TP"]
        total_fp += c["FP"]
        total_fn += c["FN"]
        total_gt += c["GT"]
        total_pred += c["Pred"]

    overall_metrics = metric_values(total_tp, total_fp, total_fn)
    overall = {
        "experiment": experiment_name,
        "conf": conf,
        "nms_iou": nms_iou,
        "match_iou": match_iou,
        "images": len(gt_by_image),
        "GT": total_gt,
        "Pred": total_pred,
        **overall_metrics,
    }
    class_rows.append({
        "experiment": experiment_name,
        "conf": conf,
        "nms_iou": nms_iou,
        "match_iou": match_iou,
        "class_id": "ALL",
        "class_name": "ALL",
        "GT": total_gt,
        "Pred": total_pred,
        **overall_metrics,
    })
    return class_rows, overall, per_image_rows + unmatched_rows


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            fieldnames = ["no_rows"]
        else:
            fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_standard_validation(model: YOLO) -> str:
    """
    标准验证：conf=0.001，避免高 conf 截断预测候选影响 AP 评估。
    这里的标准 mAP 用于统一基线，不与参数扫描的固定阈值 F1 混为一谈。
    """
    print("\n[1/2] 开始标准六类验证（conf=0.001，imgsz=640）...")
    result = model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        conf=0.001,
        iou=0.70,
        max_det=MAX_DET,
        plots=True,
        verbose=True,
        project=str(OUTPUT_ROOT),
        name="standard_validation",
        exist_ok=True,
    )
    save_dir = str(getattr(result, "save_dir", OUTPUT_ROOT / "standard_validation"))
    print(f"\n标准验证完成，结果目录：{save_dir}")
    return save_dir


def run_parameter_sweep(
    model: YOLO,
    images: List[Path],
    gt_by_image: Dict[str, List[Dict]],
) -> None:
    all_class_rows: List[Dict] = []
    all_overall_rows: List[Dict] = []
    all_per_image_rows: List[Dict] = []
    all_unmatched_rows: List[Dict] = []

    print("\n[2/2] 开始四组推理参数比较...")
    print(f"验证图片数：{len(images)}")
    print("类别顺序：", ", ".join(CLASS_NAMES))

    for exp_idx, exp in enumerate(EXPERIMENTS, start=1):
        name, conf, nms_iou = exp["name"], exp["conf"], exp["nms_iou"]
        print(
            f"\n[{exp_idx}/{len(EXPERIMENTS)}] {name}: "
            f"conf={conf:.2f}, NMS IoU={nms_iou:.2f}"
        )

        results = model.predict(
            source=[str(p) for p in images],
            imgsz=IMGSZ,
            conf=conf,
            iou=nms_iou,
            max_det=MAX_DET,
            save=False,
            verbose=False,
        )

        pred_by_image: Dict[str, List[Dict]] = {p.name: [] for p in images}
        for result in results:
            image_name = Path(result.path).name
            if image_name not in pred_by_image or result.boxes is None:
                continue
            if len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for idx, (coords, score, cls_id) in enumerate(zip(xyxy, confs, classes)):
                pred_by_image[image_name].append({
                    "pred_index": idx,
                    "class_id": int(cls_id),
                    "class_name": CLASS_NAMES[int(cls_id)] if 0 <= int(cls_id) < len(CLASS_NAMES) else str(cls_id),
                    "confidence": float(score),
                    "x1": float(coords[0]), "y1": float(coords[1]),
                    "x2": float(coords[2]), "y2": float(coords[3]),
                })

        # 保存该参数组合的所有预测框，便于以后复查，不保存图片以节省空间。
        pred_rows = []
        for image_name, preds in pred_by_image.items():
            for p in preds:
                pred_rows.append({
                    "experiment": name,
                    "conf_threshold": conf,
                    "nms_iou": nms_iou,
                    "image": image_name,
                    **p,
                })
        write_csv(
            OUTPUT_ROOT / f"predictions_{name}.csv",
            pred_rows,
            [
                "experiment", "conf_threshold", "nms_iou", "image",
                "pred_index", "class_id", "class_name", "confidence",
                "x1", "y1", "x2", "y2",
            ],
        )

        for match_iou in MATCH_IOUS:
            class_rows, overall_row, details = evaluate_config(
                predictions_by_image=pred_by_image,
                gt_by_image=gt_by_image,
                match_iou=match_iou,
                experiment_name=name,
                conf=conf,
                nms_iou=nms_iou,
            )
            all_class_rows.extend(class_rows)
            all_overall_rows.append(overall_row)

            for row in details:
                # 分离逐图指标和未匹配框明细
                if row.get("type") in {"unmatched_prediction", "unmatched_ground_truth"}:
                    all_unmatched_rows.append(row)
                else:
                    all_per_image_rows.append(row)

        print(f"完成：{name}，预测框 CSV 已保存。")

    write_csv(OUTPUT_ROOT / "per_class_metrics.csv", all_class_rows)
    write_csv(OUTPUT_ROOT / "overall_metrics.csv", all_overall_rows)
    write_csv(OUTPUT_ROOT / "per_image_metrics.csv", all_per_image_rows)
    write_csv(OUTPUT_ROOT / "unmatched_details.csv", all_unmatched_rows)

    print("\n" + "=" * 108)
    print("四组参数比较完成（以下显示匹配 IoU=0.50 的总体结果）")
    print(
        f"{'实验':24s} {'conf':>6s} {'NMS':>6s} {'GT':>6s} {'Pred':>6s} "
        f"{'TP':>6s} {'FP':>6s} {'FN':>6s} {'Precision':>11s} {'Recall':>9s} {'F1':>9s}"
    )
    for row in all_overall_rows:
        if abs(float(row["match_iou"]) - 0.50) < 1e-8:
            print(
                f"{row['experiment']:24s} {row['conf']:6.2f} {row['nms_iou']:6.2f} "
                f"{row['GT']:6d} {row['Pred']:6d} {row['TP']:6d} {row['FP']:6d} "
                f"{row['FN']:6d} {row['precision']:11.4f} {row['recall']:9.4f} {row['F1']:9.4f}"
            )

    print("\n按类别的详细指标：", OUTPUT_ROOT / "per_class_metrics.csv")
    print("总体参数对比：", OUTPUT_ROOT / "overall_metrics.csv")
    print("逐图指标：", OUTPUT_ROOT / "per_image_metrics.csv")
    print("未匹配预测/真实框：", OUTPUT_ROOT / "unmatched_details.csv")
    print("每组预测框坐标：predictions_<实验名>.csv")
    print("\n注意：这些参数扫描指标是固定阈值下的匹配 Precision/Recall/F1，不是 mAP。")


def main() -> int:
    try:
        check_paths()
        images = collect_images()
        gt_by_image, total_gt = load_ground_truths(images)
    except Exception as exc:
        print(f"[输入检查失败] {exc}")
        return 1

    print("=" * 88)
    print("NEU-DET 六类模型调优：统一验证 + 四组推理参数实验")
    print(f"模型：{MODEL_PATH}")
    print(f"配置：{DATA_YAML}")
    print(f"图片目录：{IMAGE_ROOT}")
    print(f"标签目录：{LABEL_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"图片数量：{len(images)}")
    print(f"真实标注目标数：{total_gt}")
    print(f"类别：{', '.join(CLASS_NAMES)}")
    print("=" * 88)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    try:
        standard_dir = run_standard_validation(model)
    except Exception as exc:
        print(f"\n[标准验证失败] {exc}")
        print("脚本将继续执行四组固定阈值参数比较。")
        standard_dir = "FAILED"

    try:
        run_parameter_sweep(model, images, gt_by_image)
    except Exception as exc:
        print(f"\n[参数扫描失败] {exc}")
        return 2

    readme = [
        "NEU-DET 六类模型调优实验",
        f"模型：{MODEL_PATH}",
        f"data.yaml：{DATA_YAML}",
        f"验证图片数：{len(images)}",
        f"真实标注目标数：{total_gt}",
        f"标准验证目录：{standard_dir}",
        "标准验证 conf=0.001，NMS IoU=0.70，用于标准 mAP 基线。",
        "参数扫描组合：A conf=0.30/iou=0.30；B conf=0.40/iou=0.30；C conf=0.30/iou=0.70；D conf=0.30/iou=0.45。",
        "参数扫描输出的 match_iou=0.50/0.75 是预测框与真实框的匹配阈值，不是 NMS IoU。",
        "参数扫描的 Precision/Recall/F1 为固定推理参数下的定量匹配指标，不等同于 mAP。",
        "请优先查看 per_class_metrics.csv，尤其关注 crazing 与 rolled-in_scale，同时确认其他类别没有明显退化。",
    ]
    (OUTPUT_ROOT / "README.txt").write_text("\n".join(readme), encoding="utf-8")

    print("\n全部实验结束。请上传：")
    print(f"1) {OUTPUT_ROOT / 'overall_metrics.csv'}")
    print(f"2) {OUTPUT_ROOT / 'per_class_metrics.csv'}")
    print(f"3) 标准验证目录中的 results.csv")
    print("我们将据此决定后续训练优化方向。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
