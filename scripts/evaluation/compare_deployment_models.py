from pathlib import Path
import csv
from ultralytics import YOLO

DATA = r"D:\yolov8\NEU-DET\data.yaml"
OUT = Path(r"E:\yolo_project\deployment_model_comparison")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "neu_defect_train": r"E:\yolo_project\runs\neu_defect_train\weights\best.pt",
    "neu_defect_clean": r"E:\yolo_project\runs\neu_defect_clean\weights\best.pt",
    "neu_defect_mosaic000": r"E:\yolo_project\runs\neu_defect_mosaic000\weights\best.pt",
}

rows = []

for name, weight in MODELS.items():
    if not Path(weight).is_file():
        raise FileNotFoundError(f"模型权重不存在：{weight}")

    print("\n" + "=" * 70)
    print(f"模型：{name}")
    print(f"权重：{weight}")
    print("=" * 70)

    model = YOLO(weight)
    result = model.val(
        data=DATA,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        plots=False,
        project=str(OUT),
        name=name,
        exist_ok=True,
        verbose=True,
    )

    box = result.box
    for i, class_name in enumerate(model.names.values()):
        rows.append({
            "model": name,
            "class": class_name,
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "mAP50": float(box.ap50[i]),
            "mAP50_95": float(box.ap[i].mean()),
        })

    rows.append({
        "model": name,
        "class": "all",
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    })

csv_path = OUT / "three_model_standard_metrics.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "class", "precision", "recall", "mAP50", "mAP50_95"],
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n三个模型验证完成。")
print(f"汇总 CSV：{csv_path}")
