"""Repeat the previous fine-tune with two predeclared seeds; change no other setting."""
import gc
import json
import torch
from optimize_neu import OUT, DATA, MODELS, YOLO, validate, dump


def main():
    seeds = [7, 123]
    plan = dict(seeds=[42, *seeds], initialization=str(MODELS["mosaic000"]),
                changes="Only seed changes relative to texture_ft seed42.",
                acceptance="Median mAP50-95 >= original mosaic000 + .005; report all seeds and per-class changes.",
                maximum_epochs_per_run=40, patience=12,
                independent_test=False)
    dump(OUT / "round2/stability_plan.json", plan)
    results = [json.loads((OUT / "validation/texture_ft/metrics.json").read_text(encoding="utf-8"))]
    for seed in seeds:
        name = f"mosaic000_texture_ft_seed{seed}"
        run = OUT / "training" / name
        if not (run / "weights/best.pt").exists():
            if run.exists():
                raise FileExistsError(f"Incomplete run must be inspected before reuse: {run}")
            model = YOLO(str(MODELS["mosaic000"]))
            model.train(data=str(DATA / "data.yaml"), project=str(OUT / "training"), name=name,
                        epochs=40, patience=12, batch=16, imgsz=640, device=0, workers=0,
                        seed=seed, deterministic=True, optimizer="AdamW", lr0=0.0002, lrf=0.1,
                        weight_decay=0.0005, warmup_epochs=2, cos_lr=True,
                        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15, scale=0.15, translate=0.05,
                        fliplr=0.5, flipud=0.0, degrees=0.0,
                        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                        cache=False, amp=True, plots=False, save=True,
                        exist_ok=False, resume=False, rect=False)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        result = validate(f"texture_ft_seed{seed}", run / "weights/best.pt")
        results.append(result)
        dump(OUT / "round2/stability_results_in_progress.json", results)
    import numpy as np
    baseline = json.loads((OUT / "validation/mosaic000/metrics.json").read_text(encoding="utf-8"))
    median = float(np.median([r["mAP50_95"] for r in results]))
    summary = dict(plan=plan, results=results, median_mAP50_95=median,
                   mean_mAP50=float(np.mean([r["mAP50"] for r in results])),
                   std_mAP50=float(np.std([r["mAP50"] for r in results], ddof=1)),
                   baseline_mAP50_95=baseline["mAP50_95"],
                   passed_median_gate=median >= baseline["mAP50_95"] + 0.005)
    dump(OUT / "round2/stability_summary.json", summary)
    print("STABILITY", json.dumps({k:v for k,v in summary.items() if k not in {"results", "plan"}}), flush=True)


if __name__ == "__main__":
    main()
