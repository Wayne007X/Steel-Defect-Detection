"""Reproducible NEU-DET accuracy experiments; existing project/data are read-only inputs.

Uses the installed Ultralytics validator, matching and AP implementation. All
outputs go into this checkout. Never overwrites deployed weights or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "accuracy_experiments"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from PIL import Image
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

DATA = Path(r"D:\yolov8\NEU-DET")
PROJECT = Path(r"E:\yolo_project")
NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
MODELS = {name: PROJECT / "runs" / f"neu_defect_{name}" / "weights" / "best.pt"
          for name in ("clean", "mosaic000")}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def audit():
    records, issues = [], []
    hashes = defaultdict(list)
    for split, folder in (("train", "train"), ("val", "validation")):
        for image in sorted((DATA / folder / "images").rglob("*")):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            rel = image.relative_to(DATA / folder / "images")
            label = DATA / folder / "labels" / rel.with_suffix(".txt")
            with Image.open(image) as im:
                pixels = np.asarray(im.convert("RGB"))
                shape = pixels.shape
            digest = hashlib.sha256(str(shape).encode() + pixels.tobytes()).hexdigest()
            hashes[digest].append(str(image))
            classes = []
            if not label.exists():
                issues.append({"image": str(image), "issue": "missing_label"})
            else:
                for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        row = list(map(float, line.split()))
                        assert len(row) == 5 and row[0] == int(row[0]) and 0 <= row[0] < 6
                        assert all(np.isfinite(row)) and all(0 <= x <= 1 for x in row[1:])
                        assert row[3] > 0 and row[4] > 0
                        assert row[1] - row[3]/2 >= -1e-5 and row[1] + row[3]/2 <= 1+1e-5
                        assert row[2] - row[4]/2 >= -1e-5 and row[2] + row[4]/2 <= 1+1e-5
                        classes.append(int(row[0]))
                    except (ValueError, AssertionError):
                        issues.append({"label": str(label), "line": line_no, "issue": "invalid_box"})
            records.append(dict(path=str(image), label=str(label), split=split,
                                shape=list(shape), classes=classes, pixel_sha256=digest))
    duplicates = [v for v in hashes.values() if len(v) > 1]
    leakage = [v for v in duplicates if any("\\train\\" in p for p in v)
               and any("\\validation\\" in p for p in v)]
    summary = {split: dict(images=sum(r["split"] == split for r in records),
                           boxes=sum(len(r["classes"]) for r in records if r["split"] == split),
                           class_boxes=dict(Counter(NAMES[c] for r in records if r["split"] == split for c in r["classes"])))
               for split in ("train", "val")}
    summary.update(issues=issues, exact_pixel_duplicates=duplicates, cross_split_duplicates=leakage,
                   shapes=dict(Counter(str(r["shape"]) for r in records)))
    dump(OUT / "dataset_audit.json", summary)
    dump(OUT / "dataset_manifest.json", records)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


class CaptureValidator(DetectionValidator):
    """Keep native validator behavior and capture post-NMS predictions in input coordinates."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def update_metrics(self, preds, batch):
        for si, p in enumerate(preds):
            target = self._prepare_batch(si, batch)
            self.records.append(dict(
                path=str(batch["im_file"][si]),
                boxes=p["bboxes"].detach().cpu().numpy(),
                conf=p["conf"].detach().cpu().numpy(),
                cls=p["cls"].detach().cpu().numpy(),
                gt_boxes=target["bboxes"].detach().cpu().numpy(),
                gt_cls=target["cls"].detach().cpu().numpy(),
                input_shape=tuple(batch["img"].shape[2:]),
            ))
        super().update_metrics(preds, batch)

    def get_stats(self):
        np.save(self.save_dir / "predictions.npy", np.array(self.records, dtype=object), allow_pickle=True)
        return super().get_stats()


def validate(name, weights, augment=False, imgsz=640, data_yaml=None):
    destination = OUT / "validation" / name
    if (destination / "metrics.json").exists():
        return json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
    print(f"VALIDATE {name}: {weights}, augment={augment}, size={imgsz}", flush=True)
    model = YOLO(str(weights))
    data_yaml = Path(data_yaml) if data_yaml is not None else DATA / "data.yaml"
    metrics = model.val(validator=CaptureValidator, data=str(data_yaml),
                        imgsz=imgsz, batch=8, device=0, workers=0, conf=0.001,
                        iou=0.7, rect=False, augment=augment, plots=False,
                        save_json=False, save_txt=False, project=str(OUT / "validation"),
                        name=name, exist_ok=True, verbose=False)
    result = dict(name=name, weights=str(weights), augment=augment, imgsz=imgsz, data_yaml=str(data_yaml),
                  precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                  mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                  speed=metrics.speed, per_class=metrics.summary(),
                  torch=torch.__version__, ultralytics=ultralytics.__version__,
                  settings=dict(batch=8, conf=0.001, nms_iou=0.7, rect=False, device=0))
    dump(destination / "metrics.json", result)
    print(f"RESULT {name}: AP50={result['mAP50']:.6f}, AP50_95={result['mAP50_95']:.6f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "validate"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        audit()
    else:
        results = []
        for augment in (False, True):
            for name, weights in MODELS.items():
                results.append(validate(name + ("_tta" if augment else ""), weights, augment))
        dump(OUT / "native_comparison.json", results)


if __name__ == "__main__":
    main()
