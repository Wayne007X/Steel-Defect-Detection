"""Evaluate cached predictions, small predeclared fusion comparison and OOF thresholds.

AP and matching call the same installed Ultralytics functions as native val().
Cached arrays are produced locally by optimize_neu.py; never load untrusted npy.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics.utils.metrics import ap_per_class
from optimize_neu import OUT, NAMES, CaptureValidator, dump


def load(name):
    return np.load(OUT / "validation" / name / "predictions.npy", allow_pickle=True).tolist()


def subset(record, indices):
    return {**record, **{k: record[k][indices] for k in ("boxes", "conf", "cls")}}


def nms(record, iou=0.3):
    keep = batched_nms(torch.as_tensor(record["boxes"], dtype=torch.float32),
                       torch.as_tensor(record["conf"], dtype=torch.float32),
                       torch.as_tensor(record["cls"], dtype=torch.int64), iou).numpy()
    return subset(record, keep)


def ious(box, boxes):
    inter = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(1)
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    return inter / np.maximum(areas + area - inter, 1e-9)


def fuse(a, b, iou=0.55):
    """Two-model weighted boxes; one contribution per source in each cluster.

    Single-source boxes get confidence / 2; agreement gets mean confidence.
    This is a specified two-source fusion variant, not an external WBF package.
    """
    assert a["path"] == b["path"] and a["input_shape"] == b["input_shape"]
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
            eligible = [(j, float(ious(box, cluster["box"][None])[0])) for j, cluster in enumerate(clusters)
                        if source not in cluster["sources"]]
            best = max(eligible, key=lambda r: r[1], default=(-1, 0))
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


MATCHER = CaptureValidator(args=dict(plots=False))


def stats(records):
    parts = []
    for r in records:
        p = dict(bboxes=torch.from_numpy(r["boxes"]), cls=torch.from_numpy(r["cls"]))
        g = dict(bboxes=torch.from_numpy(r["gt_boxes"]), cls=torch.from_numpy(r["gt_cls"]))
        tp = MATCHER._process_batch(p, g)["tp"]
        parts.append((tp, r["conf"], r["cls"], r["gt_cls"]))
    return [np.concatenate([p[j] for p in parts], axis=0) for j in range(4)]


def evaluate(records):
    tp, conf, cls, targets = stats(records)
    result = ap_per_class(tp, conf, cls, targets, names=dict(enumerate(NAMES)))
    _, _, p, r, f1, ap, ids = result[:7]
    return dict(mAP50=float(ap[:, 0].mean()), mAP50_95=float(ap.mean()),
                precision=float(p.mean()), recall=float(r.mean()),
                per_class=[dict(name=NAMES[int(c)], precision=float(p[j]), recall=float(r[j]),
                                mAP50=float(ap[j, 0]), mAP50_95=float(ap[j].mean())) for j, c in enumerate(ids)])


def counts(records, thresholds):
    selected = [subset(r, r["conf"] >= np.asarray(thresholds)[r["cls"].astype(int)]) for r in records]
    tp, _, cls, targets = stats(selected)
    rows = []
    for c in range(6):
        t = int(tp[cls == c, 0].sum())
        fp = int((cls == c).sum()) - t
        fn = int((targets == c).sum()) - t
        rows.append(dict(name=NAMES[c], TP=t, FP=fp, FN=fn))
    return rows


def summarize(rows):
    t, fp, fn = [sum(r[k] for r in rows) for k in ("TP", "FP", "FN")]
    p = t / max(t + fp, 1)
    r = t / max(t + fn, 1)
    return dict(TP=t, FP=fp, FN=fn, precision=p, recall=r, F1=2*p*r/max(p+r, 1e-9))


def calibrate(records, beta=1):
    # Five deterministic folds, stratified by image folder. Every held fold uses
    # thresholds fitted only on the other four folds. Models were trained and
    # selected with the original validation set, so this is NOT an independent test.
    fold_ids, by_class = {}, {}
    for i, record in enumerate(records):
        by_class.setdefault(Path(record["path"]).parent.name, []).append(i)
    rng = np.random.default_rng(20261030)
    for ids in by_class.values():
        for j, i in enumerate(rng.permutation(ids)):
            fold_ids[int(i)] = j % 5
    st = [stats([record]) for record in records]
    candidates = np.arange(0.05, 0.501, 0.025)

    def fit(indices):
        tp, conf, cls, targets = [np.concatenate([st[i][j] for i in indices], axis=0) for j in range(4)]
        chosen = []
        for c in range(6):
            total = (targets == c).sum()
            best = (-1, 0.25)
            for threshold in candidates:
                selected = (cls == c) & (conf >= threshold)
                t = tp[selected, 0].sum()
                fp = selected.sum() - t
                fn = total - t
                f = (1+beta**2)*t / max((1+beta**2)*t+beta**2*fn+fp, 1)
                if f > best[0]:
                    best = (f, float(threshold))
            chosen.append(best[1])
        return chosen

    all_rows = [dict(name=n, TP=0, FP=0, FN=0) for n in NAMES]
    folds = []
    for fold in range(5):
        train = [i for i in range(len(records)) if fold_ids[i] != fold]
        held = [i for i in range(len(records)) if fold_ids[i] == fold]
        thresholds = fit(train)
        rows = counts([records[i] for i in held], thresholds)
        for c, row in enumerate(rows):
            for key in ("TP", "FP", "FN"):
                all_rows[c][key] += row[key]
        folds.append(dict(fold=fold, thresholds=thresholds, **summarize(rows)))
    return dict(beta=beta, full_validation_fit_thresholds=fit(list(range(len(records)))),
                oof=summarize(all_rows), oof_per_class=all_rows, folds=folds)


def main():
    cache = {name: load(name) for name in ("clean", "mosaic000", "clean_tta", "mosaic000_tta")}
    results = []
    for name, records in list(cache.items()):
        metrics = evaluate(records)
        native = json.loads((OUT / "validation" / name / "metrics.json").read_text(encoding="utf-8"))
        assert abs(metrics["mAP50"] - native["mAP50"]) < 1e-7, (name, metrics, native)
        assert abs(metrics["mAP50_95"] - native["mAP50_95"]) < 1e-7
        print("NATIVE PARITY PASSED", name, flush=True)
    for suffix in ("", "_tta"):
        start = time.perf_counter()
        records = [fuse(a, b) for a, b in zip(cache["clean"+suffix], cache["mosaic000"+suffix])]
        name = "fusion" + suffix
        cache[name] = records
        np.save(OUT / (name + ".npy"), np.array(records, dtype=object), allow_pickle=True)
        dump(OUT / (name + "_overhead.json"), dict(fusion_cpu_ms_per_image=1000*(time.perf_counter()-start)/len(records)))
    # Existing deployment workpoint uses NMS .30. This secondary pass starts
    # from .70-suppressed caches, so it is explicitly a sequential-NMS experiment,
    # not claimed identical to direct raw-output NMS .30.
    for name in ("clean", "mosaic000", "clean_tta", "mosaic000_tta"):
        cache[name + "_nms030"] = [nms(r, 0.3) for r in cache[name]]
    for suffix in ("", "_tta"):
        name = "fusion" + suffix + "_nms030"
        cache[name] = [fuse(a, b) for a, b in zip(cache["clean"+suffix+"_nms030"],
                                               cache["mosaic000"+suffix+"_nms030"])]
        np.save(OUT / (name + ".npy"), np.array(cache[name], dtype=object), allow_pickle=True)
    for name, records in cache.items():
        metrics = evaluate(records)
        row = dict(name=name, **metrics,
                   fixed_020=summarize(counts(records, [0.2]*6)),
                   fixed_025=summarize(counts(records, [0.25]*6)),
                   calibration_f1=calibrate(records, 1), calibration_f2=calibrate(records, 2))
        results.append(row)
        print(name, "AP50", round(row["mAP50"], 5), "AP50_95", round(row["mAP50_95"], 5),
              "OOF F1", row["calibration_f1"]["oof"], flush=True)
    dump(OUT / "candidate_comparison.json", results)


if __name__ == "__main__":
    main()
