# -*- coding: utf-8 -*-
"""
Mosaic000-only: diagnose the difference between model.predict() and model.val()
for PyTorch vs ONNX, using Ultralytics 8.4.177's native validation matcher.

Run from E:\\yolo_project with yolo_workspace activated:
    python .\\diagnose_mosaic000_val_path.py

This script does NOT train or export anything. It only evaluates mosaic000.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ap_per_class, box_iou


# ---------------- Configuration: mosaic000 ONLY ----------------
PROJECT_ROOT = Path(r"E:\yolo_project")
DATA_ROOT = Path(r"D:\yolov8\NEU-DET")
DATA_YAML = DATA_ROOT / "data.yaml"
IMAGE_ROOT = DATA_ROOT / "validation" / "images"

MODELS = {
    "mosaic000_pytorch": PROJECT_ROOT / "runs" / "neu_defect_mosaic000" / "weights" / "best.pt",
    "mosaic000_onnx": PROJECT_ROOT / "runs" / "neu_defect_mosaic000" / "weights" / "best.onnx",
}

IMG_SIZE = 640
CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300
DEVICE = "cpu"
BATCH = 1
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)

OUT_DIR = PROJECT_ROOT / "onnx_consistency" / datetime.now().strftime(
    "mosaic000_pathcheck_%Y%m%d_%H%M%S"
)


def norm_path(path) -> str:
    return str(Path(path).resolve()).lower()


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
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = sorted(
        p for p in IMAGE_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"No validation images found: {IMAGE_ROOT}")
    return paths


def load_gt(label_path: Path, width: int, height: int):
    boxes, classes = [], []
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing validation label: {label_path}")
    text = label_path.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not text:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed label {label_path}:{line_no}: {line}")
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ])
        classes.append(cls)
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(classes, dtype=torch.int64),
    )


def extract_predict_outputs(model_key: str, weight: Path, image_paths: list[Path]):
    print(f"\n[{model_key}] model.predict(): {weight}")
    model = YOLO(str(weight))
    predictions = {}
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
        key = norm_path(result.path)
        predictions[key] = {
            "path": Path(result.path).resolve(),
            "boxes": boxes,
            "confs": confs,
            "classes": classes,
            "shape": tuple(result.orig_shape),
        }
        if idx % 60 == 0 or idx == len(image_paths):
            print(f"  predict progress {idx}/{len(image_paths)}")
    if len(predictions) != len(image_paths):
        raise RuntimeError(
            f"{model_key}: expected {len(image_paths)} predict outputs, got {len(predictions)}"
        )
    return predictions, model.names


class CapturingDetectionValidator(DetectionValidator):
    """Capture the exact post-NMS predictions consumed by native model.val()."""

    last_instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured = {}
        CapturingDetectionValidator.last_instance = self

    def update_metrics(self, preds, batch):
        # Save the predictions after the validator's own postprocess/NMS.
        # Convert boxes from letterboxed validation coordinates to original pixels.
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            boxes = predn["bboxes"].detach().clone()
            boxes = ops.scale_boxes(
                pbatch["imgsz"],
                boxes,
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            )
            image_path = Path(pbatch["im_file"]).resolve()
            key = norm_path(image_path)
            self.captured[key] = {
                "path": image_path,
                "boxes": boxes.cpu().numpy().astype(np.float32),
                "confs": predn["conf"].detach().cpu().numpy().astype(np.float32),
                "classes": predn["cls"].detach().cpu().numpy().astype(np.int64),
                "shape": tuple(pbatch["ori_shape"]),
            }
        # Let Ultralytics perform the real, authoritative metric calculation.
        return super().update_metrics(preds, batch)


def official_correct_matrix(pred_boxes, pred_cls, gt_boxes, gt_cls):
    """Use the confidence-ordered greedy matcher from Ultralytics BaseValidator."""
    n_pred = len(pred_boxes)
    niou = len(IOU_THRESHOLDS)
    correct = np.zeros((n_pred, niou), dtype=bool)
    if n_pred == 0 or len(gt_boxes) == 0:
        return correct

    iou = box_iou(gt_boxes, pred_boxes)
    correct_class = gt_cls[:, None] == pred_cls[None, :]
    iou = (iou * correct_class).cpu().numpy()
    thresholds = IOU_THRESHOLDS.cpu().numpy()

    # Prediction order is already descending confidence after Ultralytics NMS.
    matched = np.zeros((iou.shape[0], len(thresholds)), dtype=bool)
    for j in np.flatnonzero((iou >= thresholds.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, np.arange(len(thresholds))] >= thresholds
        matched[k, np.arange(len(thresholds))] |= correct[j]
    return correct


def evaluate_predictions(model_key: str, predictions: dict, image_paths: list[Path], names: dict):
    """Offline AP calculation using Ultralytics' native confidence-ordered matcher."""
    correct_parts, conf_parts, pred_cls_parts, target_cls_parts = [], [], [], []

    for idx, image_path in enumerate(image_paths, 1):
        pred = predictions[norm_path(image_path)]
        height, width = pred["shape"]
        label_path = Path(img2label_paths([str(image_path)])[0])
        gt_boxes, gt_cls = load_gt(label_path, width, height)

        pred_boxes = torch.as_tensor(pred["boxes"], dtype=torch.float32).reshape(-1, 4)
        pred_cls = torch.as_tensor(pred["classes"], dtype=torch.int64)
        correct = official_correct_matrix(pred_boxes, pred_cls, gt_boxes, gt_cls)

        correct_parts.append(correct)
        conf_parts.append(pred["confs"])
        pred_cls_parts.append(pred["classes"])
        target_cls_parts.append(gt_cls.cpu().numpy())

        if idx % 60 == 0 or idx == len(image_paths):
            print(f"  offline metric progress {idx}/{len(image_paths)}")

    correct_all = np.concatenate(correct_parts, axis=0)
    conf_all = np.concatenate(conf_parts)
    pred_cls_all = np.concatenate(pred_cls_parts)
    target_cls_all = np.concatenate(target_cls_parts)

    result = ap_per_class(
        correct_all,
        conf_all,
        pred_cls_all,
        target_cls_all,
        plot=False,
        names=names,
    )
    _, _, precision, recall, _, ap, ap_class = result[:7]

    rows = []
    for pos, cls_id in enumerate(ap_class):
        cls_id = int(cls_id)
        rows.append({
            "model": model_key,
            "class_id": cls_id,
            "class": names.get(cls_id, str(cls_id)),
            "precision": float(precision[pos]),
            "recall": float(recall[pos]),
            "mAP50": float(ap[pos, 0]),
            "mAP50_95": float(np.mean(ap[pos])),
            "gt_instances": int(np.sum(target_cls_all == cls_id)),
        })
    rows.append({
        "model": model_key,
        "class_id": "all",
        "class": "all",
        "precision": float(np.mean(precision)) if len(precision) else 0.0,
        "recall": float(np.mean(recall)) if len(recall) else 0.0,
        "mAP50": float(np.mean(ap[:, 0])) if len(ap) else 0.0,
        "mAP50_95": float(np.mean(ap)) if len(ap) else 0.0,
        "gt_instances": int(len(target_cls_all)),
    })
    return rows


def match_sets(a: dict, b: dict, iou_threshold=0.5):
    """One-to-one same-class matching for comparing two prediction paths."""
    abox = torch.as_tensor(a["boxes"], dtype=torch.float32)
    bbox = torch.as_tensor(b["boxes"], dtype=torch.float32)
    acls, bcls = a["classes"], b["classes"]
    if len(abox) == 0 or len(bbox) == 0:
        return [], len(abox), len(bbox)
    ious = box_iou(abox, bbox).cpu().numpy()
    pairs = []
    for i in range(len(abox)):
        for j in range(len(bbox)):
            if int(acls[i]) == int(bcls[j]) and ious[i, j] >= iou_threshold:
                pairs.append((float(ious[i, j]), i, j))
    pairs.sort(reverse=True)
    used_a, used_b, matches = set(), set(), []
    for iou, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matches.append((iou, i, j))
    return matches, len(abox) - len(used_a), len(bbox) - len(used_b)


def compare_predict_to_val(predict_sets: dict, val_sets: dict, model_key: str):
    rows = []
    all_ious, all_conf_diffs = [], []
    totals = {
        "predict_boxes": 0, "validator_boxes": 0, "matched": 0,
        "predict_unmatched": 0, "validator_unmatched": 0,
    }
    keys = sorted(set(predict_sets) | set(val_sets))
    for key in keys:
        a, b = predict_sets.get(key), val_sets.get(key)
        if a is None or b is None:
            raise RuntimeError(f"{model_key}: missing image in predict/val outputs: {key}")
        matches, ua, ub = match_sets(a, b)
        ious = [x[0] for x in matches]
        conf_diffs = [abs(float(a["confs"][i]) - float(b["confs"][j])) for _, i, j in matches]
        all_ious.extend(ious)
        all_conf_diffs.extend(conf_diffs)
        totals["predict_boxes"] += len(a["boxes"])
        totals["validator_boxes"] += len(b["boxes"])
        totals["matched"] += len(matches)
        totals["predict_unmatched"] += ua
        totals["validator_unmatched"] += ub
        rows.append({
            "model": model_key,
            "image": key,
            "predict_boxes": len(a["boxes"]),
            "validator_boxes": len(b["boxes"]),
            "matched_same_class_iou50": len(matches),
            "predict_unmatched": ua,
            "validator_unmatched": ub,
            "mean_matched_iou": float(np.mean(ious)) if ious else "",
            "mean_confidence_abs_diff": float(np.mean(conf_diffs)) if conf_diffs else "",
        })
    summary = {
        "model": model_key,
        "images_compared": len(keys),
        **totals,
        "matched_over_predict": totals["matched"] / totals["predict_boxes"] if totals["predict_boxes"] else 0.0,
        "matched_over_validator": totals["matched"] / totals["validator_boxes"] if totals["validator_boxes"] else 0.0,
        "mean_matched_iou": float(np.mean(all_ious)) if all_ious else "",
        "mean_confidence_abs_diff": float(np.mean(all_conf_diffs)) if all_conf_diffs else "",
    }
    return rows, summary


def builtin_metric_rows(model_key: str, result, names: dict):
    box = result.box
    rows = []
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("MOSAIC000 ONLY: predict() vs native val() path check")
    print(f"Python: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {__import__('ultralytics').__version__}")
    print(f"Images: {IMAGE_ROOT}")
    print(f"Output: {OUT_DIR}")
    print(f"Settings: imgsz={IMG_SIZE}, conf={CONF}, NMS IoU={NMS_IOU}, max_det={MAX_DET}, batch={BATCH}, device={DEVICE}")
    print("=" * 72)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {IMAGE_ROOT}")
    for key, weight in MODELS.items():
        if not weight.is_file():
            raise FileNotFoundError(f"Model not found for {key}: {weight}")

    image_paths = find_images()
    print(f"Validation images found: {len(image_paths)}")
    if len(image_paths) != 360:
        raise RuntimeError(f"Expected 360 validation images, found {len(image_paths)}. Stop to avoid comparing a changed dataset.")

    all_predict_metrics = []
    all_builtin_metrics = []
    parity_rows, parity_summaries = [], []
    run_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": __import__("ultralytics").__version__,
        "data_yaml": str(DATA_YAML),
        "image_root": str(IMAGE_ROOT),
        "image_count": len(image_paths),
        "models": {k: str(v) for k, v in MODELS.items()},
        "settings": {
            "imgsz": IMG_SIZE, "conf": CONF, "nms_iou": NMS_IOU,
            "max_det": MAX_DET, "batch": BATCH, "device": DEVICE,
        },
    }

    for model_key, weight in MODELS.items():
        predict_sets, names = extract_predict_outputs(model_key, weight, image_paths)
        all_predict_metrics.extend(
            evaluate_predictions(model_key, predict_sets, image_paths, names)
        )

        print(f"\n[{model_key}] native model.val() with prediction capture")
        model = YOLO(str(weight))
        result = model.val(
            validator=CapturingDetectionValidator,
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
        validator = CapturingDetectionValidator.last_instance
        if validator is None or len(validator.captured) != len(image_paths):
            got = 0 if validator is None else len(validator.captured)
            raise RuntimeError(
                f"{model_key}: failed to capture all native val predictions "
                f"({got}/{len(image_paths)})."
            )
        val_sets = validator.captured

        all_builtin_metrics.extend(builtin_metric_rows(model_key, result, names))
        rows, summary = compare_predict_to_val(predict_sets, val_sets, model_key)
        parity_rows.extend(rows)
        parity_summaries.append(summary)

        # Persist both sets so later checks do not require another inference run.
        for path_name, pred_sets in (("predict", predict_sets), ("validator", val_sets)):
            flat_rows = []
            for image_key, item in pred_sets.items():
                for i, (box, conf, cls_id) in enumerate(zip(item["boxes"], item["confs"], item["classes"])):
                    flat_rows.append({
                        "model": model_key,
                        "path_type": path_name,
                        "image": image_key,
                        "detection_index": i,
                        "x1": float(box[0]), "y1": float(box[1]),
                        "x2": float(box[2]), "y2": float(box[3]),
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                    })
            write_csv(OUT_DIR / f"{model_key}_{path_name}_detections.csv", flat_rows)

    write_csv(OUT_DIR / "predict_path_metrics_official_matcher.csv", all_predict_metrics)
    write_csv(OUT_DIR / "native_val_metrics.csv", all_builtin_metrics)
    write_csv(OUT_DIR / "predict_vs_native_val_per_image.csv", parity_rows)
    write_csv(OUT_DIR / "predict_vs_native_val_summary.csv", parity_summaries)

    report = [
        "MOSAIC000 ONLY: predict() vs native val() diagnosis",
        "=" * 60,
        f"Images: {len(image_paths)}",
        f"Ultralytics: {__import__('ultralytics').__version__}",
        f"Settings: imgsz={IMG_SIZE}, conf={CONF}, NMS IoU={NMS_IOU}, max_det={MAX_DET}, batch={BATCH}, device={DEVICE}",
        "",
        "Important: native_val_metrics.csv is the official model.val() result.",
        "predict_path_metrics_official_matcher.csv evaluates model.predict() detections",
        "using Ultralytics' confidence-ordered matching logic. These are not guaranteed",
        "to match model.val(), because the two APIs can use different postprocessing paths.",
        "",
        "Predict-vs-native-val detection comparison:",
    ]
    for s in parity_summaries:
        report.extend([
            f"- {s['model']}:",
            f"  images={s['images_compared']}",
            f"  predict boxes={s['predict_boxes']}; native-val boxes={s['validator_boxes']}",
            f"  matched={s['matched']}; predict unmatched={s['predict_unmatched']}; native-val unmatched={s['validator_unmatched']}",
            f"  matched/predict={s['matched_over_predict']:.6f}; matched/native-val={s['matched_over_validator']:.6f}",
            f"  mean matched IoU={s['mean_matched_iou']}; mean confidence abs diff={s['mean_confidence_abs_diff']}",
        ])
    report.extend([
        "",
        "How to interpret:",
        "1. If predict-vs-native-val boxes differ materially, the earlier metric gap comes from",
        "   comparing different postprocessing/prediction sets; use native model.val() for official metrics.",
        "2. If boxes match almost perfectly but the offline metric still differs, inspect the",
        "   offline metric aggregation against native_val_metrics.csv. The native val metric remains authoritative.",
        "3. This script only evaluates mosaic000. It does not train, export, or touch clean.",
        "",
        "Files:",
        "- predict_path_metrics_official_matcher.csv",
        "- native_val_metrics.csv",
        "- predict_vs_native_val_summary.csv",
        "- predict_vs_native_val_per_image.csv",
        "- mosaic000_pytorch_predict_detections.csv",
        "- mosaic000_pytorch_validator_detections.csv",
        "- mosaic000_onnx_predict_detections.csv",
        "- mosaic000_onnx_validator_detections.csv",
        "- run_info.json",
    ])
    (OUT_DIR / "diagnosis_report.txt").write_text("\n".join(report), encoding="utf-8")
    (OUT_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("MOSAIC000 path check completed.")
    print(f"Output directory: {OUT_DIR}")
    print("Read first: diagnosis_report.txt")
    print("Then: predict_vs_native_val_summary.csv")
    print("Official metrics: native_val_metrics.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
