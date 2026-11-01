"""Apply the same two predeclared NMS profiles to every replication seed."""
import gc
import json
import numpy as np
import torch
from optimize_neu import OUT, validate, dump
from compare_predictions import load, nms, evaluate, counts, summarize
from compare_adaptive_nms import adaptive_nms
from precision_detector import fuse


def main():
    stability=json.loads((OUT / "round2/stability_summary.json").read_text(encoding="utf-8"))
    results=[]
    for seed in (42,7,123):
        base="texture_ft" if seed ==42 else f"texture_ft_seed{seed}"
        weight=OUT / f"training/mosaic000_texture_ft_seed{seed}/weights/best.pt"
        validate(base+"_tta",weight,True)
        for augment in (False,True):
            suffix="_tta" if augment else ""
            clean,texture=load("clean"+suffix),load(base+suffix)
            modes=("standard","adaptive") if augment else ("native_nms070",)
            for mode in modes:
                transform=(adaptive_nms if mode =="adaptive" else
                           (lambda r:nms(r,.3)) if mode =="standard" else lambda r:r)
                records=[fuse(transform(a),transform(b)) for a,b in zip(clean,texture)]
                per_class=counts(records,[.25]*6)
                row=dict(seed=seed,mode=mode,augment=augment,**evaluate(records),
                         fixed_025=summarize(per_class),fixed_025_per_class=per_class,
                         weak_macro_recall=float(np.mean([r["TP"]/(r["TP"]+r["FN"]) for r in per_class if r["name"] in {"crazing","rolled-in_scale"}])))
                results.append(row)
                print(seed,mode,row["mAP50"],row["mAP50_95"],row["fixed_025"],flush=True)
        gc.collect();torch.cuda.empty_cache()
    best_single=max(stability["results"],key=lambda r:r["mAP50_95"])
    seed=42 if best_single["name"]=="texture_ft" else int(best_single["name"].replace("texture_ft_seed",""))
    selected=[r for r in results if r["seed"]==seed]
    changes=[]
    for s in (42,7,123):
        standard=next(r for r in results if r["seed"]==s and r["mode"]=="standard")
        adaptive=next(r for r in results if r["seed"]==s and r["mode"]=="adaptive")
        changes.append(dict(seed=s,mAP50=adaptive["mAP50"]-standard["mAP50"],
                            weak_macro_recall=adaptive["weak_macro_recall"]-standard["weak_macro_recall"],
                            F1=adaptive["fixed_025"]["F1"]-standard["fixed_025"]["F1"]))
    dump(OUT / "round2/seed_fusion_comparison.json",dict(results=results,
         single_selection="Highest native mAP50-95 among the three seeds; exploratory validation selection.",
         selected_single_seed=seed,selected_single=best_single,selected_seed_fusion=selected,
         adaptive_minus_standard_by_seed=changes,
         median_adaptive_AP50_gain=float(np.median([r["mAP50"] for r in changes]))))
    print("SELECTED SINGLE SEED",seed,flush=True)


if __name__ == "__main__":
    main()
