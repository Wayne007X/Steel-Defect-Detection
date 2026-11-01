"""Read-only source-label audit and reproducible error review for weak classes.

Diagnostic labels describe overlap/score patterns; they are not claims of
annotation mistakes or physical causes. Original data is never edited.
"""
from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from optimize_neu import OUT, DATA, NAMES, dump
from compare_predictions import MATCHER, stats
import torch

DEST = OUT / "round2/error_audit"
WEAK = {0, 4}
CONF = 0.25
FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH.exists() else ImageFont.load_default()


def overlap(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    aa = np.maximum(0, a[:, 2:] - a[:, :2]).prod(1)
    bb = np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)
    return inter / np.maximum(aa[:, None] + bb[None] - inter, 1e-9)


def assignments(r, conf=CONF):
    """Match at IoU .5 in score order, asserting agreement with native matcher."""
    keep = np.flatnonzero(r["conf"] >= conf)
    keep = keep[np.argsort(-r["conf"][keep], kind="stable")]
    matrix = overlap(r["gt_boxes"], r["boxes"])
    claimed, matches = set(), {}
    for i in keep:
        eligible = [j for j in range(len(r["gt_cls"]))
                    if j not in claimed and r["gt_cls"][j] == r["cls"][i]]
        if eligible:
            j = max(eligible, key=lambda j: matrix[j, i])
            if matrix[j, i] >= 0.5:
                matches[int(i)] = j
                claimed.add(j)
    p = dict(bboxes=torch.as_tensor(r["boxes"][keep]), cls=torch.as_tensor(r["cls"][keep]))
    g = dict(bboxes=torch.as_tensor(r["gt_boxes"]), cls=torch.as_tensor(r["gt_cls"]))
    expected = int(MATCHER._process_batch(p, g)["tp"][:, 0].sum())
    assert len(matches) == expected
    return keep, matches, claimed, matrix


def classify(r):
    keep, matches, claimed, matrix = assignments(r)
    rows = []
    for j, c in enumerate(r["gt_cls"].astype(int)):
        if j in claimed:
            continue
        same = np.flatnonzero(r["cls"] == c)
        wrong = np.flatnonzero(r["cls"] != c)
        above = same[r["conf"][same] >= CONF]
        low = same[r["conf"][same] < CONF]
        best_iou = float(matrix[j, same].max()) if len(same) else 0.
        low_hits = low[matrix[j, low] >= 0.5]
        if len(above) and matrix[j, above].max() >= 0.5:
            reason = "competing_ground_truth"
        elif len(low_hits):
            reason = "low_confidence_candidate"
        elif len(above) and matrix[j, above].max() >= 0.1:
            reason = "localization_at_workpoint"
        elif len(wrong) and matrix[j, wrong].max() >= 0.5:
            reason = "class_confusion_candidate"
        elif best_iou >= 0.1:
            reason = "low_confidence_and_localization"
        else:
            reason = "no_overlapping_candidate"
        best = same[np.argmax(matrix[j, same])] if len(same) else None
        best_low = low_hits[np.argmax(r["conf"][low_hits])] if len(low_hits) else None
        rows.append(dict(path=r["path"], class_name=NAMES[c], class_id=int(c), kind="FN", target_index=j,
                         reason=reason, best_same_class_iou=best_iou,
                         best_overlap_conf=float(r["conf"][best]) if best is not None else 0.,
                         highest_matching_low_conf=float(r["conf"][best_low]) if best_low is not None else None))
    for i in keep:
        if int(i) in matches:
            continue
        c = int(r["cls"][i])
        same = np.flatnonzero(r["gt_cls"] == c)
        wrong = np.flatnonzero(r["gt_cls"] != c)
        best_iou = float(matrix[same, i].max()) if len(same) else 0.
        if best_iou >= 0.5:
            reason = "duplicate_or_competing_prediction"
        elif best_iou >= 0.1:
            reason = "localization_overlap"
        elif len(wrong) and matrix[wrong, i].max() >= 0.5:
            reason = "class_confusion_candidate"
        else:
            reason = "no_same_class_gt_overlap_review_required"
        rows.append(dict(path=r["path"], class_name=NAMES[c], class_id=c, kind="FP", prediction_index=int(i),
                         reason=reason, best_same_class_iou=best_iou, confidence=float(r["conf"][i])))
    return rows, keep, matches, claimed


def xml_and_near_duplicate_audit():
    manifest = json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
    xml_issues, missing_xml, gray, hashes, descriptors = [], [], [], [], []
    box_sizes = defaultdict(list)
    for rec in manifest:
        path = Path(rec["path"])
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(small)[:8, :8].ravel()[1:]
        hashes.append(dct > np.median(dct))
        gray.append(cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel())
        descriptors.append(dict(path=str(path), split=rec["split"], image_class=path.parent.name,
                                mean_brightness=float(image.mean()), std_contrast=float(image.std())))
        xml_path = DATA / ("train" if rec["split"] == "train" else "validation") / "annotations" / (path.stem + ".xml")
        yolo_rows = [list(map(float, line.split())) for line in Path(rec["label"]).read_text().splitlines() if line.strip()]
        for row in yolo_rows:
            box_sizes[(rec["split"], int(row[0]))].append([row[3]*image.shape[1], row[4]*image.shape[0], row[3]*row[4]])
        if not xml_path.is_file():
            missing_xml.append(str(xml_path))
            continue
        root = ET.parse(xml_path).getroot()
        w = int(root.findtext("size/width"))
        h = int(root.findtext("size/height"))
        expected = []
        for obj in root.findall("object"):
            name = obj.findtext("name").strip()
            box = obj.find("bndbox")
            x1, y1, x2, y2 = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            expected.append([NAMES.index(name), (x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h])
        expected, actual = np.array(sorted(expected)), np.array(sorted(yolo_rows))
        if expected.shape != actual.shape or not np.allclose(expected, actual, atol=0.000001, rtol=0):
            xml_issues.append(dict(image=str(path), xml=str(xml_path), issue="XML_YOLO_disagreement"))
        if (h, w) != image.shape:
            xml_issues.append(dict(image=str(path), issue="XML_image_size_disagreement"))
    hashes, gray = np.array(hashes), np.array(gray)
    gray -= gray.mean(1, keepdims=True)
    gray /= np.maximum(np.linalg.norm(gray, axis=1, keepdims=True), 1e-9)
    nearest = []
    for i, desc in enumerate(descriptors):
        if desc["split"] != "val":
            continue
        eligible = [j for j, d in enumerate(descriptors) if d["split"] == "train" and d["image_class"] == desc["image_class"]]
        distances = np.count_nonzero(hashes[eligible] != hashes[i], axis=1)
        best = int(np.argmin(distances))
        j = eligible[best]
        nearest.append(dict(val=desc["path"], train=descriptors[j]["path"],
                            phash_hamming=int(distances[best]), aligned_correlation=float(gray[i] @ gray[j])))
    sizes = []
    for (split, c), values in box_sizes.items():
        arr = np.array(values)
        sizes.append(dict(split=split, class_name=NAMES[c], boxes=len(arr),
                          median_width_px=float(np.median(arr[:, 0])), median_height_px=float(np.median(arr[:, 1])),
                          median_area_fraction=float(np.median(arr[:, 2])),
                          small_area_lt_32px_square=int((arr[:, 0]*arr[:, 1] < 32**2).sum())))
    summary = dict(xml_checked=len(manifest)-len(missing_xml), missing_xml=missing_xml, xml_issues=xml_issues,
                   near_duplicate_candidates=[r for r in nearest if r["phash_hamming"] <= 6],
                   note="pHash candidates are similarity-screening results, not proof of duplicate images or shared acquisition batches.",
                   box_sizes=sizes)
    dump(DEST / "source_label_and_similarity_audit.json", summary)
    dump(DEST / "image_descriptors.json", descriptors)
    dump(DEST / "nearest_train_per_validation_image.json", nearest)
    return summary


def annotate(image, boxes, labels, colors, title, size=320):
    panel = Image.new("RGB", (size, size+35), "white")
    panel.paste(image.resize((size, size), Image.Resampling.NEAREST), (0, 35))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), title, fill="black", font=font(16))
    sx, sy = size/image.width, size/image.height
    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = box
        points = (max(0,x1*sx), max(35,y1*sy+35), min(size-1,x2*sx), min(size+34,y2*sy+35))
        draw.rectangle(points, outline=color, width=2)
        x, y = points[0], max(35, points[1]-16)
        text_box = draw.textbbox((x, y), label, font=font(13))
        draw.rectangle(text_box, fill="black")
        draw.text((x, y), label, fill=color, font=font(13))
    return panel


def draw_review(r, errors, keep, matches, claimed):
    with Image.open(r["path"]) as source:
        image = source.convert("RGB")
    scale = np.array([image.width/640, image.height/640]*2)
    gboxes = r["gt_boxes"]*scale
    boxes = r["boxes"][keep]*scale
    gtlabels = [f"G{j} {NAMES[int(c)]}" for j, c in enumerate(r["gt_cls"])]
    glabels = [("TP " if j in claimed else "FN ")+label for j, label in enumerate(gtlabels)]
    plabels = [("TP " if int(i) in matches else "FP ")+f"{r['conf'][i]:.2f}" for i in keep]
    panels = [annotate(image, [], [], [], "Original"),
              annotate(image, gboxes, glabels, ["lime" if j in claimed else "red" for j in range(len(gboxes))], "GT: red=missed at conf .25"),
              annotate(image, boxes, plabels, ["cyan" if int(i) in matches else "orange" for i in keep], "Prediction: orange=FP")]
    canvas = Image.new("RGB", (984, 420), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 2), Path(r["path"]).name, fill="black", font=font(17))
    for j, panel in enumerate(panels):
        canvas.paste(panel, (j*328, 28))
    counts = Counter(e["kind"] for e in errors)
    text = f"TP={len(matches)} FN={counts['FN']} FP={counts['FP']} | " + ", ".join(sorted({e["reason"] for e in errors}))
    draw.text((5, 390), text[:135], fill="black", font=font(12))
    return canvas


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "images").mkdir(exist_ok=True)
    (DEST / "contact_sheets").mkdir(exist_ok=True)
    source_summary = xml_and_near_duplicate_audit()
    records = np.load(OUT / "final_deployable_pt_tta_nms030_predictions.npy", allow_pickle=True).tolist()
    errors, summaries, reviews = [], [], defaultdict(list)
    for r in records:
        rows, keep, matches, claimed = classify(r)
        errors.extend(rows)
        counts = Counter(e["kind"] for e in rows)
        summaries.append(dict(path=r["path"], image_class=Path(r["path"]).parent.name,
                              TP=len(matches), FP=counts["FP"], FN=counts["FN"],
                              GT=len(r["gt_cls"]), predictions=len(keep)))
        if any(int(c) in WEAK for c in r["gt_cls"]):
            image = draw_review(r, rows, keep, matches, claimed)
            image.save(DEST / "images" / (Path(r["path"]).stem+".png"))
            reviews[Path(r["path"]).parent.name].append((counts["FN"]*3+counts["FP"], Path(r["path"]).stem))
    for name, images in reviews.items():
        images.sort(key=lambda row: (-row[0], row[1]))
        for start in range(0, len(images), 12):
            sheet = Image.new("RGB", (1968, 1260), "#dddddd")
            for j, (_, stem) in enumerate(images[start:start+12]):
                with Image.open(DEST / "images" / (stem+".png")) as image:
                    thumb = image.resize((656, 280), Image.Resampling.LANCZOS)
                    sheet.paste(thumb, ((j%3)*656, (j//3)*315))
            sheet.save(DEST / "contact_sheets" / f"{name}_{start//12+1:02}.jpg", quality=92)
    weak_summary = {}
    for c in sorted(WEAK):
        weak_summary[NAMES[c]] = {kind:dict(Counter(e["reason"] for e in errors if e["class_id"] == c and e["kind"] == kind))
                                  for kind in ("FN", "FP")}
    summary = dict(configuration=dict(model="clean+texture_seed42 TTA", conf=CONF, nms_iou=0.3, match_iou=0.5),
                   images=len(records), weak_images=sum(map(len,reviews.values())),
                   totals={key:sum(r[key] for r in summaries) for key in ("TP", "FP", "FN", "GT")},
                   weak_class_errors=weak_summary, source_label_audit=source_summary,
                   diagnostic_warning="Reasons are overlap/score patterns; semantic annotation correctness requires review.")
    dump(DEST / "summary.json", summary)
    dump(DEST / "error_details.json", errors)
    dump(DEST / "per_image.json", summaries)
    for name, rows in (("error_details", errors), ("per_image", summaries)):
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (DEST / (name+".csv")).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k != "source_label_audit"}, ensure_ascii=False), flush=True)
    print("XML issues",len(source_summary["xml_issues"]),"similarity candidates",len(source_summary["near_duplicate_candidates"]), flush=True)


if __name__ == "__main__":
    main()
