"""Bounded fine-tune: preserve subtle grayscale texture, keep deployed models untouched."""
import json
from optimize_neu import ROOT, OUT, DATA, MODELS, YOLO, validate, dump


def main():
    name = "mosaic000_texture_ft_seed42"
    run = OUT / "training" / name
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite previous experiment: {run}")
    # A single augmentation-profile hypothesis, not a pure single-parameter ablation.
    # The old lr0005 experiment started from COCO weights; this starts from the
    # already-trained mosaic000 checkpoint and uses a small fine-tuning rate.
    plan = dict(
        hypothesis="Gentler scale/brightness perturbations preserve fine grayscale defects during fine-tuning.",
        init=str(MODELS["mosaic000"]),
        acceptance="mAP50-95 must exceed both baselines by 0.005; no class AP50 loss >0.03 vs mosaic000.",
        max_epochs=40, patience=12, seed=42,
        limitation="Same repeatedly-used validation set; improvement requires independent-data confirmation.",
    )
    dump(OUT / "texture_experiment_plan.json", plan)
    model = YOLO(str(MODELS["mosaic000"]))
    model.train(data=str(DATA / "data.yaml"), project=str(OUT / "training"), name=name,
                epochs=40, patience=12, batch=16, imgsz=640, device=0, workers=0,
                seed=42, deterministic=True, optimizer="AdamW", lr0=0.0002, lrf=0.1,
                weight_decay=0.0005, warmup_epochs=2, cos_lr=True,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.15, scale=0.15, translate=0.05,
                fliplr=0.5, flipud=0.0, degrees=0.0,
                mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                cache=False, amp=True, plots=False, save=True,
                exist_ok=False, resume=False, rect=False)
    weights = run / "weights" / "best.pt"
    metrics = validate("texture_ft", weights)
    baselines = [json.loads((OUT / "validation" / n / "metrics.json").read_text(encoding="utf-8"))
                 for n in ("clean", "mosaic000")]
    base_class = {r["Class"]: r["mAP50"] for r in baselines[1]["per_class"]}
    # Current Ultralytics summary uses mAP50; fail clearly if its schema changes.
    class_losses = {r["Class"]: float(r["mAP50"]) - float(base_class[r["Class"]])
                    for r in metrics["per_class"]}
    accepted = metrics["mAP50_95"] >= max(b["mAP50_95"] for b in baselines) + 0.005
    accepted = accepted and min(class_losses.values()) >= -0.03
    dump(OUT / "texture_acceptance.json", dict(accepted=accepted, per_class_AP50_changes=class_losses,
                                                metrics=metrics, plan=plan))
    print("ACCEPTED", accepted, flush=True)


if __name__ == "__main__":
    main()
