"""Two-model accuracy detector compatible with the existing API's YOLO.predict interface.

Supports local YOLOv8 .pt (optionally TTA) or raw-output .onnx checkpoints.
NMS .70 -> optional second NMS -> two-source box fusion is deliberate: it
matches the measured cached pipeline, including candidate floor and max_det.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


def ious(box, boxes):
    inter = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(1)
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    return inter / np.maximum(areas + area - inter, 1e-9)


def fuse(a, b, iou=0.55):
    """Confidence-weighted geometry; at most one box from each model per cluster."""
    assert a["path"] == b["path"] and a["input_shape"] == b["input_shape"]
    if "gt_boxes" in a:
        assert np.array_equal(a["gt_boxes"], b["gt_boxes"])
    boxes, scores, classes = [], [], []
    for c in range(6):
        groups = []
        for source, record in enumerate((a, b)):
            ids = np.flatnonzero((record["cls"] == c) & (record["conf"] >= 0.01))
            groups.extend((float(record["conf"][i]), source, record["boxes"][i]) for i in ids)
        groups.sort(key=lambda r: r[0], reverse=True)
        clusters = []
        for score, source, box in groups:
            eligible = [j for j, cluster in enumerate(clusters) if source not in cluster["sources"]]
            if eligible:
                overlaps = ious(box, np.stack([clusters[j]["box"] for j in eligible]))
                index = int(np.argmax(overlaps))
                best = (eligible[index], float(overlaps[index]))
            else:
                best = (-1, 0)
            if best[1] >= iou:
                cluster = clusters[best[0]]
                cluster["weighted"] += score * box
                cluster["score"] += score
                cluster["box"] = cluster["weighted"] / cluster["score"]
                cluster["sources"].add(source)
            else:
                clusters.append(dict(box=box.copy(), weighted=score * box, score=score, sources={source}))
        for cluster in clusters:
            boxes.append(cluster["box"])
            scores.append(cluster["score"] / 2)
            classes.append(c)
    order = np.argsort(-np.asarray(scores), kind="stable")[:300]
    return {**a, "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4)[order],
            "conf": np.asarray(scores, dtype=np.float32)[order],
            "cls": np.asarray(classes, dtype=np.float32)[order]}


class PrecisionDetector:
    def __init__(self, weights, *, device="cpu", augment=False, thresholds=None, class_nms=None):
        if len(weights) not in (1, 2):
            raise ValueError("One or two checkpoints are required")
        self.device = torch.device(device)
        self.augment = augment
        self.thresholds = thresholds
        if class_nms is not None:
            if len(class_nms) != 6 or not all(0 < t <= 0.7 for t in class_nms):
                raise ValueError("Expected six class NMS thresholds in (0, .7]")
        self.class_nms = class_nms
        self.backends = []
        self.names = None
        for checkpoint in weights:
            path = Path(checkpoint)
            if path.suffix.lower() == ".onnx":
                if augment:
                    raise ValueError("TTA requires .pt checkpoints; use augment=False with ONNX")
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                metadata = session.get_modelmeta().custom_metadata_map
                names = {int(k): v for k, v in ast.literal_eval(metadata["names"]).items()}
                self.backends.append(("onnx", session))
            elif path.suffix.lower() == ".pt":
                model = YOLO(str(path)).model.to(self.device).float().eval()
                model.fuse(verbose=False)
                names = model.names
                self.backends.append(("pt", model))
            else:
                raise ValueError("Expected .pt or .onnx checkpoint")
            if self.names is not None and self.names != names:
                raise ValueError("Checkpoint class names/order differ")
            self.names = names
        if len(self.names) != 6:
            raise ValueError("This experiment expects the six NEU-DET classes")

    @torch.inference_mode()
    def records(self, source, *, imgsz=640, iou=0.7):
        if isinstance(source, Image.Image):
            original = np.asarray(source.convert("RGB"))[:, :, ::-1].copy()
        elif isinstance(source, (str, Path)):
            original = cv2.imread(str(source))
            if original is None:
                raise ValueError(f"Cannot decode {source}")
        else:
            original = np.asarray(source)
        if original.ndim != 3 or original.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR array, RGB PIL image, or image path")
        if imgsz != 640:
            raise ValueError("This validated detector uses imgsz=640")
        image = LetterBox(new_shape=(imgsz, imgsz), auto=False)(image=original)
        array = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        tensor = torch.from_numpy(array).to(self.device)
        records = []
        for kind, backend in self.backends:
            if kind == "pt":
                output = backend(tensor, augment=self.augment)[0]
            else:
                output = torch.from_numpy(backend.run(None, {backend.get_inputs()[0].name: array})[0])
            detection = nms.non_max_suppression(output, conf_thres=0.001, iou_thres=0.7,
                                               multi_label=True, max_det=300)[0].cpu()
            if self.class_nms is not None:
                parts = []
                for c, threshold in enumerate(self.class_nms):
                    part = detection[detection[:, 5] == c]
                    if threshold < 0.7:
                        keep = batched_nms(part[:, :4], part[:, 4], part[:, 5].long(), threshold)
                        part = part[keep]
                    parts.append(part)
                detection = torch.cat(parts)
                order = np.argsort(-detection[:, 4].numpy(), kind="stable")
                detection = detection[torch.as_tensor(order)]
            elif iou < 0.7:
                keep = batched_nms(detection[:, :4], detection[:, 4], detection[:, 5].long(), iou)
                detection = detection[keep]
            records.append(dict(path=str(source) if isinstance(source, (str, Path)) else "memory",
                                input_shape=(imgsz, imgsz), boxes=detection[:, :4].numpy(),
                                conf=detection[:, 4].numpy(), cls=detection[:, 5].numpy()))
        return original, records[0] if len(records) == 1 else fuse(*records)

    def predict(self, source, imgsz=640, conf=0.25, iou=0.7, max_det=300, device=None, verbose=False, **kwargs):
        original, record = self.records(source, imgsz=imgsz, iou=iou)
        thresholds = (np.asarray(self.thresholds)[record["cls"].astype(int)]
                      if self.thresholds is not None else conf)
        selected = record["conf"] >= thresholds
        boxes = torch.from_numpy(record["boxes"][selected].copy())
        boxes = ops.scale_boxes((imgsz, imgsz), boxes, original.shape[:2])
        detection = torch.cat((boxes, torch.from_numpy(record["conf"][selected, None]),
                               torch.from_numpy(record["cls"][selected, None])), dim=1)[:max_det]
        return [Results(original, path=str(source) if isinstance(source, (str, Path)) else "memory",
                        names=self.names, boxes=detection)]
