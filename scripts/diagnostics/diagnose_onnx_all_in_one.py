# -*- coding: utf-8 -*-
"""
One-shot PyTorch vs ONNX diagnostic for the NEU-DET clean and mosaic000 models.

What it does:
1. Runs all four model/backend combinations on the same 360 validation images.
2. Uses batch=1, CPU, imgsz/conf/NMS settings consistently.
3. Compares final detection boxes and confidence scores per image.
4. Evaluates all four prediction sets with ONE shared Ultralytics AP evaluator.
5. Runs Ultralytics' built-in validation for all four combinations using batch=1.
6. Writes CSV/JSON/TXT summaries to a timestamped output folder.

Run from E:\\yolo_project with the yolo_workspace environment activated:
    python .\\diagnose_onnx_all_in_one.py
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths
from ultralytics.utils.metrics import ap_per_class, box_iou


# --------------------------- Configuration ---------------------------
PROJECT_ROOT = Path(r"E:\yolo_project")
DATA_ROOT = Path(r"D:\yolov8\NEU-DET")
DATA_YAML = DATA_ROOT / "data.yaml"
IMAGE_ROOT = DATA_ROOT / "validation" / "images"

MODELS = {
    "clean_pytorch": PROJECT_ROOT / "runs" / "neu_defect_clean" / "weights" / "best.pt",
    "clean_onnx": PROJECT_ROOT / "runs" / "neu_defect_clean" / "weights" / "best.onnx",
    "mosaic000_pytorch": PROJECT_ROOT / "runs" / "neu_defect_mosaic000" / "weights" / "best.pt",
    "mosaic000_onnx": PROJECT_ROOT / "runs" / "neu_defect_mosaic000" / "weights" / "best.onnx",
}

IMG_SIZE = 640
CONF = 0.001
NMS_IOU = 0.7
MATCH_IOU = 0.5
MAX_DET = 300
DEVICE = "cpu"
BATCH = 1  # Important: ONNX here has a static batch-1 input.
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)

OUT_ROOT = PROJECT_ROOT / "onnx_consistency"
RUN_ID = datetime.now().strftime("diagnosis_%Y%m%d_%H%M%S")
OUT_DIR = OUT_ROOT / RUN_ID


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No rows\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_images() -> list[Path]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = sorted(
        p for p in IMAGE_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_ext
    )
    if not paths:
        raise FileNotFoundError(f"No validation images found: {IMAGE_ROOT}")
    return paths


def load_yolo_label(label_path: Path, width: int, height: int):
    """Return GT xyxy boxes in original-image pixels and integer class IDs."""
    boxes, classes = [], []
    if not label_path.is_file():
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )
    text = label_path.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not text:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed label {label_path}:{line_no}: {line}")
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        boxes.append([x1, y1, x2, y2])
        classes.append(cls)
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(classes, dtype=torch.int64)


def extract_predictions(model_key: str, weight_path: Path, image_paths: list[Path]):
    print(f"\n[{model_key}] predict: {weight_path}")
    model = YOLO(str(weight_path))
    names = model.names

    predictions = {}
    # stream=True limits memory; batch=1 is deliberately identical for every backend.
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        batch=BATCH,
        device=DEVICE,
        stream=True,
        verbose=False,
        save=False,
    )

    for idx, result in enumerate(results, 1):
        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            classes = np.empty((0,), dtype=np.int64)
        else:
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = result.boxes.conf.cpu().numpy().astype(np.float32)
            classes = result.boxes.cls.cpu().numpy().astype(np.int64)

        image_path = Path(result.path).resolve()
        predictions[str(image_path).lower()] = {
            "path": image_path,
            "boxes": boxes,
            "confs": confs,
            "classes": classes,
            "shape": tuple(result.orig_shape),  # height, width
            "names": names,
        }
        if idx % 60 == 0 or idx == len(image_paths):
            print(f"  progress {idx}/{len(image_paths)}")

    if len(predictions) != len(image_paths):
        raise RuntimeError(
            f"{model_key}: expected {len(image_paths)} predictions, got {len(predictions)}"
        )
    return predictions, names


def greedy_parity_match(a: dict, b: dict, threshold: float = MATCH_IOU):
    """Match same-class detections by descending IoU, one-to-one."""
    abox = torch.as_tensor(a["boxes"], dtype=torch.float32)
    bbox = torch.as_tensor(b["boxes"], dtype=torch.float32)
    acls, bcls = a["classes"], b["classes"]
    if len(abox) == 0 or len(bbox) == 0:
        return [], len(abox), len(bbox)

    ious = box_iou(abox, bbox).cpu().numpy()
    pairs = []
    for i in range(len(abox)):
        for j in range(len(bbox)):
            if int(acls[i]) == int(bcls[j]) and ious[i, j] >= threshold:
                pairs.append((float(ious[i, j]), i, j))
    pairs.sort(reverse=True)

    used_a, used_b, matches = set(), set(), []
    for iou_value, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matches.append((iou_value, i, j))
    return matches, len(abox) - len(used_a), len(bbox) - len(used_b)


def compare_prediction_sets(pred_a: dict, pred_b: dict, key_a: str, key_b: str):
    keys = sorted(set(pred_a) | set(pred_b))
    rows = []
    all_ious, all_conf_diffs = [], []
    total_a = total_b = total_matched = unmatched_a = unmatched_b = 0

    for image_key in keys:
        a = pred_a.get(image_key)
        b = pred_b.get(image_key)
        if a is None or b is None:
            rows.append({
                "image": image_key, "model_a": key_a, "model_b": key_b,
                "boxes_a": len(a["boxes"]) if a else 0,
                "boxes_b": len(b["boxes"]) if b else 0,
                "matched_same_class_iou50": 0,
                "unmatched_a": len(a["boxes"]) if a else 0,
                "unmatched_b": len(b["boxes"]) if b else 0,
                "mean_matched_iou": "",
                "mean_confidence_abs_diff": "",
            })
            continue

        matches, ua, ub = greedy_parity_match(a, b)
        ious = [m[0] for m in matches]
        conf_diffs = [
            abs(float(a["confs"][i]) - float(b["confs"][j]))
            for _, i, j in matches
        ]
        all_ious.extend(ious)
        all_conf_diffs.extend(conf_diffs)
        total_a += len(a["boxes"])
        total_b += len(b["boxes"])
        total_matched += len(matches)
        unmatched_a += ua
        unmatched_b += ub
        rows.append({
            "image": image_key,
            "model_a": key_a,
            "model_b": key_b,
            "boxes_a": len(a["boxes"]),
            "boxes_b": len(b["boxes"]),
            "matched_same_class_iou50": len(matches),
            "unmatched_a": ua,
            "unmatched_b": ub,
            "mean_matched_iou": float(np.mean(ious)) if ious else "",
            "mean_confidence_abs_diff": float(np.mean(conf_diffs)) if conf_diffs else "",
        })

    summary = {
        "model_a": key_a,
        "model_b": key_b,
        "images_compared": len(keys),
        "boxes_a": total_a,
        "boxes_b": total_b,
        "matched_boxes": total_matched,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
        "matched_over_a": total_matched / total_a if total_a else 0.0,
        "matched_over_b": total_matched / total_b if total_b else 0.0,
        "mean_matched_iou": float(np.mean(all_ious)) if all_ious else "",
        "mean_confidence_abs_diff": float(np.mean(all_conf_diffs)) if all_conf_diffs else "",
    }
    return rows, summary


def shared_ap_evaluation(model_key: str, predictions: dict, image_paths: list[Path], names: dict):
    """
    Compute AP from the already-decoded predict() detections using the same
    evaluator for every backend. This deliberately avoids model.val() backend
    differences while calculating the shared metric.
    """
    print(f"\n[{model_key}] shared AP evaluation")
    niou = len(IOU_THRESHOLDS)
    correct_parts, conf_parts, pred_cls_parts, target_cls_parts = [], [], [], []
    missing_labels = []

    for idx, image_path in enumerate(image_paths, 1):
        key = str(image_path.resolve()).lower()
        pred = predictions[key]
        height, width = pred["shape"]
        label_path = Path(img2label_paths([str(image_path)])[0])
        if not label_path.exists():
            missing_labels.append(str(label_path))

        gt_boxes, gt_cls = load_yolo_label(label_path, width, height)
        pred_boxes = torch.as_tensor(pred["boxes"], dtype=torch.float32).reshape(-1, 4)
        pred_conf = pred["confs"].astype(np.float32)
        pred_cls = pred["classes"].astype(np.int64)

        correct = torch.zeros((len(pred_boxes), niou), dtype=torch.bool)
        if len(pred_boxes) and len(gt_boxes):
            iou = box_iou(gt_boxes, pred_boxes)
            correct_class = gt_cls[:, None] == torch.as_tensor(pred_cls)[None, :]
            for t_idx, threshold in enumerate(IOU_THRESHOLDS):
                gt_i, pred_i = torch.where((iou >= threshold) & correct_class)
                if gt_i.numel():
                    matches = torch.stack(
                        (gt_i, pred_i, iou[gt_i, pred_i]), dim=1
                    ).cpu().numpy()
                    if len(matches) > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), t_idx] = True

        correct_parts.append(correct.cpu().numpy())
        conf_parts.append(pred_conf)
        pred_cls_parts.append(pred_cls)
        target_cls_parts.append(gt_cls.cpu().numpy())

        if idx % 60 == 0 or idx == len(image_paths):
            print(f"  progress {idx}/{len(image_paths)}")

    if missing_labels:
        preview = "\n".join(missing_labels[:10])
        raise FileNotFoundError(
            f"Missing label files ({len(missing_labels)}). First paths:\n{preview}\n"
            "Check that validation/images and validation/labels mirror each other."
        )

    correct_all = np.concatenate(correct_parts, axis=0) if correct_parts else np.zeros((0, niou), bool)
    conf_all = np.concatenate(conf_parts) if conf_parts else np.zeros((0,), np.float32)
    pred_cls_all = np.concatenate(pred_cls_parts) if pred_cls_parts else np.zeros((0,), np.int64)
    target_cls_all = np.concatenate(target_cls_parts) if target_cls_parts else np.zeros((0,), np.int64)

    # Ultralytics' own AP implementation, applied identically to all four prediction sets.
    ap_result = ap_per_class(
        correct_all,
        conf_all,
        pred_cls_all,
        target_cls_all,
        plot=False,
        names=names,
    )
    tp, fp, precision, recall, f1, ap, ap_class = ap_result[:7]

    rows = []
    for pos, cls_id in enumerate(ap_class):
        cls_id = int(cls_id)
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        rows.append({
            "model": model_key,
            "class_id": cls_id,
            "class": cls_name,
            "precision": float(precision[pos]),
            "recall": float(recall[pos]),
            "mAP50": float(ap[pos, 0]),
            "mAP50_95": float(np.mean(ap[pos])),
            "gt_instances": int(np.sum(target_cls_all == cls_id)),
        })

    if len(ap):
        rows.append({
            "model": model_key,
            "class_id": "all",
            "class": "all",
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "mAP50": float(np.mean(ap[:, 0])),
            "mAP50_95": float(np.mean(ap)),
            "gt_instances": int(len(target_cls_all)),
        })
    return rows


def builtin_validation(model_key: str, weight_path: Path, names: dict):
    """Run built-in val with matching CPU, image size, confidence, NMS and batch."""
    print(f"\n[{model_key}] built-in val: batch={BATCH}, device={DEVICE}")
    model = YOLO(str(weight_path))
    result = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=0,
        conf=CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        plots=False,
        project=str(OUT_DIR / "builtin_val"),
        name=model_key,
        exist_ok=True,
        verbose=False,
    )

    rows = []
    box = result.box
    for cls_id, cls_name in names.items():
        rows.append({
            "model": model_key,
            "class_id": int(cls_id),
            "class": str(cls_name),
            "precision": float(box.p[int(cls_id)]),
            "recall": float(box.r[int(cls_id)]),
            "mAP50": float(box.ap50[int(cls_id)]),
            "mAP50_95": float(np.mean(box.ap[int(cls_id)])),
        })
    rows.append({
        "model": model_key,
        "class_id": "all",
        "class": "all",
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    })
    return rows


def get_all_row(rows: list[dict], model_key: str):
    for row in rows:
        if row.get("model") == model_key and row.get("class") == "all":
            return row
    return None


def main():
    print("=" * 72)
    print("NEU-DET PyTorch / ONNX 一次性诊断")
    print(f"Python: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {__import__('ultralytics').__version__}")
    print(f"Images: {IMAGE_ROOT}")
    print(f"Output: {OUT_DIR}")
    print(f"Settings: imgsz={IMG_SIZE}, conf={CONF}, NMS IoU={NMS_IOU}, batch={BATCH}, device={DEVICE}")
    print("=" * 72)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {IMAGE_ROOT}")
    for key, weight in MODELS.items():
        if not weight.is_file():
            raise FileNotFoundError(f"Model not found for {key}: {weight}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = find_images()
    print(f"Validation images found: {len(image_paths)}")
    if len(image_paths) != 360:
        print(f"WARNING: expected 360 images based on prior run; found {len(image_paths)}.")

    run_info = {
        "run_id": RUN_ID,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": __import__("ultralytics").__version__,
        "onnxruntime": None,
        "data_yaml": str(DATA_YAML),
        "image_root": str(IMAGE_ROOT),
        "image_count": len(image_paths),
        "models": {k: str(v) for k, v in MODELS.items()},
        "settings": {
            "imgsz": IMG_SIZE, "conf": CONF, "nms_iou": NMS_IOU,
            "match_iou": MATCH_IOU, "max_det": MAX_DET,
            "device": DEVICE, "batch": BATCH,
        },
    }
    try:
        import onnxruntime as ort
        run_info["onnxruntime"] = ort.__version__
        run_info["onnxruntime_providers"] = ort.get_available_providers()
    except Exception as exc:
        run_info["onnxruntime_error"] = str(exc)

    # Predict all candidates once using exactly the same inference settings.
    prediction_sets, model_names = {}, {}
    for key, weight in MODELS.items():
        prediction_sets[key], model_names[key] = extract_predictions(key, weight, image_paths)

    # Detection parity: same underlying trained model, PyTorch vs ONNX.
    parity_rows, parity_summaries = [], []
    for pt_key, onnx_key in [
        ("clean_pytorch", "clean_onnx"),
        ("mosaic000_pytorch", "mosaic000_onnx"),
    ]:
        rows, summary = compare_prediction_sets(
            prediction_sets[pt_key], prediction_sets[onnx_key], pt_key, onnx_key
        )
        parity_rows.extend(rows)
        parity_summaries.append(summary)
    write_csv(OUT_DIR / "detection_parity_per_image.csv", parity_rows)
    write_csv(OUT_DIR / "detection_parity_summary.csv", parity_summaries)

    # One common evaluator for all four prediction sets.
    shared_rows = []
    for key in MODELS:
        shared_rows.extend(
            shared_ap_evaluation(key, prediction_sets[key], image_paths, model_names[key])
        )
    write_csv(OUT_DIR / "shared_evaluator_metrics.csv", shared_rows)

    # Native validator, now all batch=1 and CPU.
    builtin_rows = []
    for key, weight in MODELS.items():
        builtin_rows.extend(builtin_validation(key, weight, model_names[key]))
    write_csv(OUT_DIR / "ultralytics_builtin_metrics.csv", builtin_rows)

    # Compare shared-evaluator and built-in metrics for each candidate/backend.
    diagnosis_rows = []
    for key in MODELS:
        shared_all = get_all_row(shared_rows, key)
        builtin_all = get_all_row(builtin_rows, key)
        if shared_all and builtin_all:
            diagnosis_rows.append({
                "model": key,
                "shared_precision": shared_all["precision"],
                "builtin_precision": builtin_all["precision"],
                "precision_abs_diff": abs(shared_all["precision"] - builtin_all["precision"]),
                "shared_recall": shared_all["recall"],
                "builtin_recall": builtin_all["recall"],
                "recall_abs_diff": abs(shared_all["recall"] - builtin_all["recall"]),
                "shared_mAP50": shared_all["mAP50"],
                "builtin_mAP50": builtin_all["mAP50"],
                "mAP50_abs_diff": abs(shared_all["mAP50"] - builtin_all["mAP50"]),
                "shared_mAP50_95": shared_all["mAP50_95"],
                "builtin_mAP50_95": builtin_all["mAP50_95"],
                "mAP50_95_abs_diff": abs(shared_all["mAP50_95"] - builtin_all["mAP50_95"]),
            })
    write_csv(OUT_DIR / "evaluator_crosscheck.csv", diagnosis_rows)

    # A concise human-readable report with no unsupported automatic root-cause claim.
    report_lines = [
        "NEU-DET PyTorch / ONNX diagnosis",
        "=" * 50,
        f"Images: {len(image_paths)}",
        f"Settings: imgsz={IMG_SIZE}, conf={CONF}, NMS IoU={NMS_IOU}, batch={BATCH}, device={DEVICE}",
        "",
        "1) Detection parity (PyTorch vs ONNX)",
    ]
    for s in parity_summaries:
        report_lines.extend([
            f"- {s['model_a']} vs {s['model_b']}",
            f"  matched={s['matched_boxes']}; unmatched A={s['unmatched_a']}; unmatched B={s['unmatched_b']}",
            f"  mean matched IoU={s['mean_matched_iou']}; mean confidence abs diff={s['mean_confidence_abs_diff']}",
        ])
    report_lines.extend(["", "2) Shared evaluator versus Ultralytics built-in validation"])
    for row in diagnosis_rows:
        report_lines.extend([
            f"- {row['model']}:",
            f"  shared mAP50={row['shared_mAP50']:.6f}; builtin mAP50={row['builtin_mAP50']:.6f}; abs diff={row['mAP50_abs_diff']:.6f}",
            f"  shared mAP50-95={row['shared_mAP50_95']:.6f}; builtin mAP50-95={row['builtin_mAP50_95']:.6f}; abs diff={row['mAP50_95_abs_diff']:.6f}",
        ])
    report_lines.extend([
        "",
        "Interpretation guide:",
        "- First compare clean_pytorch with clean_onnx, then mosaic000_pytorch with mosaic000_onnx in shared_evaluator_metrics.csv.",
        "- If shared-evaluator metrics are close but built-in metrics differ, focus on backend-specific validation/preprocessing/postprocessing.",
        "- If shared-evaluator metrics also differ, inspect confidence ranking and unmatched detections in detection_parity_per_image.csv.",
        "- This run forces batch=1 for every backend; prior runs used PyTorch batch=4 while ONNX was forced to batch=1, so those earlier built-in validation numbers were not a fully controlled comparison.",
        "- Do not treat CPU PyTorch vs CPU ONNX timing as a guaranteed deployment-speed result.",
        "",
        "Files:",
        "- detection_parity_per_image.csv",
        "- detection_parity_summary.csv",
        "- shared_evaluator_metrics.csv",
        "- ultralytics_builtin_metrics.csv",
        "- evaluator_crosscheck.csv",
        "- run_info.json",
    ])
    (OUT_DIR / "diagnosis_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    (OUT_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("诊断完成。结果目录：")
    print(OUT_DIR)
    print("优先查看：")
    print("  diagnosis_report.txt")
    print("  detection_parity_summary.csv")
    print("  shared_evaluator_metrics.csv")
    print("  ultralytics_builtin_metrics.csv")
    print("  evaluator_crosscheck.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
