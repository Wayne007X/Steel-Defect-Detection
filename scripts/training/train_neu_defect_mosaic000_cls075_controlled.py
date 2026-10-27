# -*- coding: utf-8 -*-
"""
一次受控的 YOLOv8n cls=0.75 微调实验。
- 从 neu_defect_mosaic000 的 best.pt 权重开始
- 保持 imgsz=640、mosaic=0、batch=16、epochs=100、patience=20 等主要设置
- 只把 cls 从原来的 0.5 改为 0.75
- 训练前对基线做一次统一标准验证；训练后用相同设置验证新 best.pt
- 不覆盖旧模型，不使用 Cursor Agent
"""

from pathlib import Path
import csv
import sys
import traceback

from ultralytics import YOLO

# ======================== 路径 ========================
PROJECT_ROOT = Path(r"E:\yolo_project")
RUNS_ROOT = PROJECT_ROOT / "runs"
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")

BASELINE_WEIGHT = RUNS_ROOT / "neu_defect_mosaic000" / "weights" / "best.pt"

TRAIN_NAME = "neu_defect_mosaic000_cls075_controlled"
BASELINE_VAL_NAME = "neu_defect_mosaic000_cls075_baseline_fair_val"
CANDIDATE_VAL_NAME = "neu_defect_mosaic000_cls075_fair_val"
SUMMARY_CSV = RUNS_ROOT / "neu_defect_mosaic000_cls075_comparison.csv"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# ======================== 训练参数 ========================
EPOCHS = 100
PATIENCE = 20
BATCH = 16
IMGSZ = 640
DEVICE = 0
WORKERS = 0
SEED = 42

# ======================== 统一验证参数 ========================
VAL_BATCH = 4
VAL_CONF = 0.001
VAL_IOU = 0.70
MAX_DET = 300


def check_paths():
    missing = []
    if not DATA_YAML.is_file():
        missing.append(str(DATA_YAML))
    if not BASELINE_WEIGHT.is_file():
        missing.append(str(BASELINE_WEIGHT))
    if not RUNS_ROOT.is_dir():
        missing.append(str(RUNS_ROOT))
    if missing:
        raise FileNotFoundError("以下必需路径不存在：\n" + "\n".join(missing))

    # 避免覆盖已有实验或结果。
    planned = [
        RUNS_ROOT / TRAIN_NAME,
        RUNS_ROOT / BASELINE_VAL_NAME,
        RUNS_ROOT / CANDIDATE_VAL_NAME,
        SUMMARY_CSV,
    ]
    existing = [str(p) for p in planned if p.exists()]
    if existing:
        raise FileExistsError(
            "为避免覆盖已有结果，脚本停止。以下路径已经存在：\n"
            + "\n".join(existing)
            + "\n如需重新运行，请修改脚本顶部对应的名称。"
        )


def extract_metric_rows(metrics, model_label):
    """提取 Ultralytics 标准验证结果，兼容六类数据集。"""
    box = metrics.box
    rows = []

    rows.append({
        "model": model_label,
        "class_id": "all",
        "class_name": "all",
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
        "mAP50": round(float(box.map50), 6),
        "mAP50_95": round(float(box.map), 6),
    })

    class_indices = list(getattr(box, "ap_class_index", []))
    p_values = list(getattr(box, "p", []))
    r_values = list(getattr(box, "r", []))
    ap50_values = list(getattr(box, "ap50", []))
    ap_values = list(getattr(box, "ap", []))

    for class_id, class_name in enumerate(CLASS_NAMES):
        if class_id in class_indices:
            idx = class_indices.index(class_id)
            precision = float(p_values[idx]) if idx < len(p_values) else 0.0
            recall = float(r_values[idx]) if idx < len(r_values) else 0.0
            ap50 = float(ap50_values[idx]) if idx < len(ap50_values) else 0.0
            ap = float(ap_values[idx]) if idx < len(ap_values) else 0.0
        else:
            precision = recall = ap50 = ap = 0.0

        rows.append({
            "model": model_label,
            "class_id": class_id,
            "class_name": class_name,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "mAP50": round(ap50, 6),
            "mAP50_95": round(ap, 6),
        })
    return rows


def write_summary(rows):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "class_id", "class_name", "precision", "recall", "mAP50", "mAP50_95"]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_weight(weight_path, run_name, model_label):
    print("\n" + "=" * 78)
    print(f"统一标准验证：{model_label}")
    print(f"权重：{weight_path}")
    print(
        f"data={DATA_YAML}, imgsz={IMGSZ}, batch={VAL_BATCH}, "
        f"conf={VAL_CONF}, iou={VAL_IOU}, max_det={MAX_DET}"
    )
    print("=" * 78)

    val_model = YOLO(str(weight_path))
    metrics = val_model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMGSZ,
        batch=VAL_BATCH,
        device=DEVICE,
        workers=WORKERS,
        conf=VAL_CONF,
        iou=VAL_IOU,
        max_det=MAX_DET,
        plots=True,
        save_json=False,
        project=str(RUNS_ROOT),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )
    rows = extract_metric_rows(metrics, model_label)
    del val_model
    return rows


def main():
    check_paths()

    print("=" * 78)
    print("YOLOv8n 受控微调：cls 0.50 -> 0.75")
    print("起始权重：", BASELINE_WEIGHT)
    print("数据配置：", DATA_YAML)
    print("训练输出：", RUNS_ROOT / TRAIN_NAME)
    print("训练参数：epochs=100, patience=20, batch=16, imgsz=640, mosaic=0.0, cls=0.75")
    print("说明：只训练一次；旧权重不会被覆盖。")
    print("=" * 78)

    # 1. 先用统一参数重新验证基线，保证对照条件一致。
    all_rows = validate_weight(
        BASELINE_WEIGHT,
        BASELINE_VAL_NAME,
        "neu_defect_mosaic000_baseline",
    )
    write_summary(all_rows)
    print(f"\n基线指标已保存：{SUMMARY_CSV}")

    # 2. 从 mosaic000 的 best.pt 继续微调。只改变 cls=0.75。
    model = YOLO(str(BASELINE_WEIGHT))
    train_results = model.train(
        data=str(DATA_YAML),
        task="detect",
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_ROOT),
        name=TRAIN_NAME,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=SEED,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        cache=False,
        val=True,
        split="val",
        save=True,
        save_period=-1,
        plots=True,
        conf=None,
        iou=VAL_IOU,
        max_det=MAX_DET,
        save_json=False,
        mosaic=0.0,
        cls=0.75,
    )
    del model

    best_weight = RUNS_ROOT / TRAIN_NAME / "weights" / "best.pt"
    if not best_weight.is_file():
        raise FileNotFoundError(
            "训练结束后没有找到 best.pt：\n"
            + str(best_weight)
            + "\n请检查训练日志和实际输出目录。"
        )

    print("\n训练完成，最佳权重：", best_weight)

    # 3. 使用与基线完全相同的验证参数评估新权重。
    candidate_rows = validate_weight(
        best_weight,
        CANDIDATE_VAL_NAME,
        "neu_defect_mosaic000_cls075",
    )
    all_rows.extend(candidate_rows)
    write_summary(all_rows)

    print("\n" + "=" * 78)
    print("实验流程完成。")
    print("训练记录：", RUNS_ROOT / TRAIN_NAME / "results.csv")
    print("新权重：", best_weight)
    print("基线验证目录：", RUNS_ROOT / BASELINE_VAL_NAME)
    print("新模型验证目录：", RUNS_ROOT / CANDIDATE_VAL_NAME)
    print("六类统一对比 CSV：", SUMMARY_CSV)
    print("=" * 78)
    print("判断时请同时检查：整体 mAP50、整体 mAP50-95、crazing 指标、rolled-in_scale 指标及其余四类是否退步。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n脚本运行失败：")
        print(exc)
        traceback.print_exc()
        sys.exit(1)
