"""Final bounded comparison: two existing baselines paired with the new fine-tune."""
import numpy as np
from optimize_neu import OUT, dump
from compare_predictions import load, nms, evaluate, counts, summarize, calibrate
from precision_detector import fuse


def main():
    results = []
    for tta in (False, True):
        suffix = "_tta" if tta else ""
        texture = load("texture_ft" + suffix)
        for base in ("clean", "mosaic000"):
            records = load(base + suffix)
            for threshold in (0.7, 0.3):
                fused = [fuse(a if threshold == 0.7 else nms(a, threshold),
                              b if threshold == 0.7 else nms(b, threshold)) for a, b in zip(records, texture)]
                name = f"{base}_texture{suffix}_nms{int(threshold*100):03}"
                result = dict(name=name, base=base, texture="texture_ft", augment=tta, nms_iou=threshold,
                              **evaluate(fused), fixed_025=summarize(counts(fused, [0.25]*6)),
                              fixed_020=summarize(counts(fused, [0.2]*6)))
                results.append(result)
                np.save(OUT / (name + ".npy"), np.array(fused, dtype=object), allow_pickle=True)
                print(name, result["mAP50"], result["mAP50_95"], result["fixed_025"], flush=True)
    dump(OUT / "texture_fusion_comparison.json", results)
    # No further parameter search. Record calibration only for the two
    # complementary final winners, not used as a new model-selection criterion.
    winners = {max(results, key=lambda r:r[metric])["name"] for metric in ("mAP50", "mAP50_95")}
    for name in winners:
        records = np.load(OUT / (name+".npy"), allow_pickle=True).tolist()
        dump(OUT / (name+"_threshold_calibration.json"), calibrate(records, 1))


if __name__ == "__main__":
    main()
