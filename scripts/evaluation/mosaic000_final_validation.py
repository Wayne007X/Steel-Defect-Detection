# -*- coding: utf-8 -*-
"""
MOSAIC000 final, apples-to-apples native validation.
- Only evaluates mosaic000 best.pt and best.onnx.
- No training, no export, no custom metric implementation, no predict-vs-val box capture.
- Explicit rect=False for BOTH backends so PyTorch does not use rectangular batches
  while the exported ONNX backend is forced to square/non-rectangular validation.
Run:
    python .\mosaic000_final_validation.py
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

PROJECT_ROOT = Path(r"E:\yolo_project")
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
WEIGHTS = {
    "mosaic000_pytorch": PROJECT_ROOT / "runs" / "neu_defect_mosaic000" / "weights" / "best.pt",
    "mosaic000_onnx": PROJECT_ROOT / "runs" / "neu_defect_mosaic000" / "weights" / "best.onnx",
}
IMG_SIZE = 640
CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300
BATCH = 1
DEVICE = "cpu"
WORKERS = 0
RECT = False  # Critical: force same square preprocessing policy for both backends.

OUT_DIR = PROJECT_ROOT / "onnx_consistency" / datetime.now().strftime("mosaic000_final_val_%Y%m%d_%H%M%S")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    print("=" * 76)
    print("MOSAIC000 FINAL VALIDATION — native Ultralytics val() only")
    print(f"Python: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Output: {OUT_DIR}")
    print(f"Settings: imgsz={IMG_SIZE}, conf={CONF}, iou={NMS_IOU}, max_det={MAX_DET}, "
          f"batch={BATCH}, device={DEVICE}, rect={RECT}, workers={WORKERS}")
    print("=" * 76)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
    for name, path in WEIGHTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file ({name}): {path}")

    rows = []
    raw_results = {}
    for name, weight in WEIGHTS.items():
        print(f"\n[{name}] Native model.val(): {weight}")
        model = YOLO(str(weight))
        result = model.val(
            data=str(DATA_YAML),
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            conf=CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            rect=RECT,
            plots=False,
            save_json=False,
            project=str(OUT_DIR / "native_val"),
            name=name,
            exist_ok=True,
            verbose=True,
        )
        box = result.box
        raw_results[name] = {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
        }
        for class_id, class_name in model.names.items():
            rows.append({
                "model": name,
                "class_id": int(class_id),
                "class": str(class_name),
                "precision": float(box.p[int(class_id)]),
                "recall": float(box.r[int(class_id)]),
                "mAP50": float(box.ap50[int(class_id)]),
                "mAP50_95": float(box.ap[int(class_id)].mean()),
            })
        rows.append({
            "model": name,
            "class_id": "all",
            "class": "all",
            **raw_results[name],
        })

    csv_path = OUT_DIR / "native_validation_metrics_rect_false.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pt = raw_results["mosaic000_pytorch"]
    ox = raw_results["mosaic000_onnx"]
    comparison = {
        "pytorch": pt,
        "onnx": ox,
        "onnx_minus_pytorch": {k: ox[k] - pt[k] for k in pt},
        "settings": {
            "imgsz": IMG_SIZE, "conf": CONF, "nms_iou": NMS_IOU,
            "max_det": MAX_DET, "batch": BATCH, "device": DEVICE,
            "rect": RECT, "workers": WORKERS,
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "weights": {k: str(v) for k, v in WEIGHTS.items()},
            "data_yaml": str(DATA_YAML),
        },
        "note": (
            "Native Ultralytics model.val() metrics. rect=False was explicitly applied "
            "to both models to avoid PyTorch rectangular validation versus exported-model "
            "square validation preprocessing. This is a fairer backend comparison; it does "
            "not promise identical metrics because postprocessing/backend implementations "
            "can still differ."
        ),
    }
    json_path = OUT_DIR / "native_validation_comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = OUT_DIR / "FINAL_REPORT.txt"
    lines = [
        "MOSAIC000 FINAL NATIVE VALIDATION",
        "=" * 60,
        f"Ultralytics: {ultralytics.__version__}",
        f"Settings: imgsz={IMG_SIZE}, conf={CONF}, iou={NMS_IOU}, max_det={MAX_DET}, "
        f"batch={BATCH}, device={DEVICE}, rect={RECT}",
        "",
        "Official native model.val() aggregate metrics:",
    ]
    for key in ("mosaic000_pytorch", "mosaic000_onnx"):
        m = raw_results[key]
        lines.append(f"{key}: P={m['precision']:.6f}, R={m['recall']:.6f}, "
                     f"mAP50={m['mAP50']:.6f}, mAP50-95={m['mAP50_95']:.6f}")
    lines += [
        "",
        "ONNX minus PyTorch:",
        ", ".join(f"{k}={v:+.6f}" for k, v in comparison["onnx_minus_pytorch"].items()),
        "",
        "Important: rect=False is explicitly applied to both backends. Earlier runs allowed "
        "PyTorch validation to use rect=True by default, while exported ONNX validation uses "
        "rect=False for fixed-shape exported models. That made the prior backend comparison "
        "not fully controlled.",
        "",
        f"Metrics CSV: {csv_path}",
        f"Comparison JSON: {json_path}",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "=" * 76)
    print("FINAL VALIDATION COMPLETED")
    print(f"Report: {report_path}")
    print(f"Metrics: {csv_path}")
    print(f"Comparison: {json_path}")
    print("This run only validates mosaic000; no training/export performed.")
    print("=" * 76)


if __name__ == "__main__":
    main()
