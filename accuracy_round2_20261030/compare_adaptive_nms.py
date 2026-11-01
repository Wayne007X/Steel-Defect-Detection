"""Evidence-driven inference experiment: retain overlapping weak-class proposals.

Hypothesis fixed before evaluation: NMS .50 for crazing/rolled-in_scale, .30
for the other classes; fusion unchanged. No parameter sweep or label edits.
"""
import numpy as np
from optimize_neu import OUT, NAMES, dump
from compare_predictions import load, nms, subset, evaluate, counts, summarize, calibrate
from precision_detector import fuse


def adaptive_nms(record):
    parts = []
    for c in range(6):
        part = subset(record, record["cls"] == c)
        parts.append(nms(part, 0.5 if c in {0, 4} else 0.3))
    combined = {**record, **{k:np.concatenate([r[k] for r in parts]) for k in ("boxes", "conf", "cls")}}
    order = np.argsort(-combined["conf"], kind="stable")
    return subset(combined, order)


def main():
    destination = OUT / "round2"
    dump(destination / "adaptive_nms_plan.json", dict(
        hypothesis="Adjacent weak-class defects may be suppressed at NMS .30; relax weak classes to .50.",
        class_nms=[0.5,0.3,0.3,0.3,0.5,0.3], conf=0.25,
        unchanged="clean+texture seed42, native TTA, fusion IoU .55, score aggregation and geometry",
        gate="mAP50 +.003, weak-class macro recall +.02, overall F1 loss no worse than .005; no class AP50 loss >.02",
        warning="Exploratory comparison on repeatedly-used validation data, not independent confirmation."))
    clean, texture = load("clean_tta"), load("texture_ft_tta")
    baseline = [fuse(nms(a, 0.3), nms(b, 0.3)) for a, b in zip(clean, texture)]
    candidate = [fuse(adaptive_nms(a), adaptive_nms(b)) for a, b in zip(clean, texture)]
    results = []
    for name, records in (("global_nms030", baseline), ("adaptive_weak050", candidate)):
        per_class = counts(records, [0.25]*6)
        row = dict(name=name, **evaluate(records), fixed_025=summarize(per_class),
                   fixed_025_per_class=per_class, threshold_calibration=calibrate(records, 1))
        row["weak_macro_recall"] = float(np.mean([r["TP"]/(r["TP"]+r["FN"]) for r in per_class if r["name"] in {NAMES[0], NAMES[4]}]))
        results.append(row)
        np.save(destination / (name+".npy"), np.array(records,dtype=object), allow_pickle=True)
        print(name, row["mAP50"], row["mAP50_95"], row["fixed_025"], "weak recall", row["weak_macro_recall"], flush=True)
    baseline, candidate = results
    class_changes = {c["name"]:c["mAP50"]-b["mAP50"] for b,c in zip(baseline["per_class"],candidate["per_class"])}
    accepted = (candidate["mAP50"] >= baseline["mAP50"]+.003
                and candidate["weak_macro_recall"] >= baseline["weak_macro_recall"]+.02
                and candidate["fixed_025"]["F1"] >= baseline["fixed_025"]["F1"]-.005
                and min(class_changes.values()) >= -.02)
    dump(destination / "adaptive_nms_comparison.json", dict(accepted=accepted, results=results, class_AP50_changes=class_changes))
    print("ACCEPTED",accepted, flush=True)


if __name__ == "__main__":
    main()
