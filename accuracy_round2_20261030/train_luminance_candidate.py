"""One predeclared brightness-shift experiment, all artifacts in an isolated copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml
from optimize_neu import OUT, NAMES, MODELS, YOLO, validate, dump
from luminance_normalization import normalize_luminance

DATASET=OUT / "round2/normalized_dataset"


def prepare():
    if (DATASET / "manifest.json").exists():
        return
    if DATASET.exists():
        raise FileExistsError("Inspect incomplete normalized dataset before reuse")
    manifest=json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
    outputs=[]
    for row in manifest:
        source=Path(row["path"])
        split="train" if row["split"]=="train" else "validation"
        target=DATASET / split / "images" / source.parent.name / (source.stem+".png")
        target_label=DATASET / split / "labels" / source.parent.name / (source.stem+".txt")
        target.parent.mkdir(parents=True,exist_ok=True)
        target_label.parent.mkdir(parents=True,exist_ok=True)
        image=cv2.imread(str(source))
        normalized=normalize_luminance(image)
        assert image.shape==normalized.shape
        assert cv2.imwrite(str(target),normalized)
        shutil.copy2(row["label"],target_label)
        assert target_label.read_bytes()==Path(row["label"]).read_bytes()
        outputs.append(dict(original=str(source),normalized=str(target),split=row["split"],
                            source_pixel_sha256=row["pixel_sha256"],
                            normalized_file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            label_sha256=hashlib.sha256(target_label.read_bytes()).hexdigest(),
                            original_mean=float(image.mean()),normalized_mean=float(normalized.mean())))
    (DATASET / "data.yaml").write_text(yaml.safe_dump(dict(path=str(DATASET),train="train/images",val="validation/images",nc=6,names=NAMES),sort_keys=False),encoding="utf-8")
    dump(DATASET / "manifest.json",outputs)
    shift={}
    for cls in ("crazing","rolled-in_scale"):
        shift[cls]={}
        for split in ("train","val"):
            selected=[r for r in outputs if r["split"]==split and Path(r["original"]).parent.name==cls]
            shift[cls][split]={key:float(np.mean([r[key] for r in selected])) for key in ("original_mean","normalized_mean")}
    dump(OUT / "round2/normalization_shift_check.json",shift)
    print("PREPARED",len(outputs),json.dumps(shift),flush=True)


def train():
    prepare()
    name="mosaic000_texture_normalized_seed42"
    run=OUT / "training" / name
    if run.exists():
        raise FileExistsError(run)
    plan=dict(hypothesis="Per-image grayscale normalization reduces the observed train/val brightness shift, especially rolled-in_scale.",
              method="Map each image's own 1st and 99th grayscale percentiles to 0/255; clip outside, preserve coordinates.",
              source_labels_edited=False,dataset_statistics_fitted=False,
              initialization=str(MODELS["mosaic000"]),seed=42,maximum_epochs=40,patience=12,
              control="Same texture fine-tune profile/optimizer as prior seed42; preprocessing is the changed factor.",
              gate="Weak-class mean AP50 improves >=.03 over texture seed42; overall AP50 >= control+.003, AP50-95 >= control-.005; no other-class AP50 loss >.03.",
              limitation="Same validation images, deterministic different preprocessing; exploratory, not an independent test.")
    dump(OUT / "round2/normalization_experiment_plan.json",plan)
    model=YOLO(str(MODELS["mosaic000"]))
    model.train(data=str(DATASET / "data.yaml"), project=str(OUT / "training"), name=name,
                epochs=40, patience=12, batch=16, imgsz=640, device=0, workers=0,
                seed=42, deterministic=True, optimizer="AdamW", lr0=.0002, lrf=.1,
                weight_decay=.0005, warmup_epochs=2, cos_lr=True,
                hsv_h=0., hsv_s=0., hsv_v=.15, scale=.15, translate=.05,
                fliplr=.5, flipud=0., degrees=0.,mosaic=0., mixup=0.,cutmix=0.,copy_paste=0.,
                cache=False, amp=True, plots=False, save=True,exist_ok=False,resume=False,rect=False)
    result=validate("texture_normalized",run / "weights/best.pt",data_yaml=DATASET / "data.yaml")
    control=json.loads((OUT / "validation/texture_ft/metrics.json").read_text(encoding="utf-8"))
    changes={c["Class"]:c["mAP50"]-b["mAP50"] for b,c in zip(control["per_class"],result["per_class"])}
    weak_gain=float(np.mean([changes[n] for n in ("crazing","rolled-in_scale")]))
    passed=(weak_gain>=.03 and result["mAP50"]>=control["mAP50"]+.003
            and result["mAP50_95"]>=control["mAP50_95"]-.005
            and min(v for k,v in changes.items() if k not in {"crazing","rolled-in_scale"})>=-.03)
    dump(OUT / "round2/normalization_acceptance.json",dict(accepted=passed,metrics=result,
          control=control,class_AP50_changes=changes,weak_mean_AP50_gain=weak_gain,plan=plan))
    print("NORMALIZATION ACCEPTED",passed,"weak AP50 gain",weak_gain,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","train"])
    args=parser.parse_args()
    prepare() if args.stage=="prepare" else train()
