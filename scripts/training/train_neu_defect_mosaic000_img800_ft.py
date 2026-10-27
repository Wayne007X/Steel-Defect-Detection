# -*- coding: utf-8 -*-
"""NEU-DET YOLOv8n: 640 -> 800 short fine-tuning + matched validation.

Run in the existing Windows yolo_workspace environment; no Cursor Agent required.
The script trains from the existing mosaic000 best.pt, then validates:
1) baseline checkpoint at imgsz=640
2) baseline checkpoint at imgsz=800
3) fine-tuned checkpoint at imgsz=800
"""

from pathlib import Path
import sys
import traceback
from ultralytics import YOLO

BASE_MODEL = Path(r"E:\yolo_project\runs\neu_defect_mosaic000\weights\best.pt")
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
RUNS_DIR = Path(r"E:\yolo_project\runs")

TRAIN_NAME = "neu_defect_mosaic000_img800_ft"
VAL_BASE_640 = "neu_defect_mosaic000_baseline_img640_val"
VAL_BASE_800 = "neu_defect_mosaic000_baseline_img800_val"
VAL_FT_800 = "neu_defect_mosaic000_img800_ft_val"

EPOCHS = 30
PATIENCE = 10
IMGSZ = 800
BATCH = 4
VAL_CONF = 0.001
VAL_IOU = 0.7
MAX_DET = 300


def validate(model_path: Path, imgsz: int, run_name: str) -> None:
    print(f"\n{'=' * 72}\n验证：{run_name}\n权重：{model_path}\nimgsz={imgsz}\n{'=' * 72}")
    YOLO(str(model_path)).val(
        data=str(DATA_YAML),
        split="val",
        imgsz=imgsz,
        batch=BATCH,
        device=0,
        workers=0,
        conf=VAL_CONF,
        iou=VAL_IOU,
        max_det=MAX_DET,
        plots=True,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
    )


def main() -> int:
    for p in (BASE_MODEL, DATA_YAML):
        if not p.is_file():
            print(f"找不到文件：{p}")
            return 1

    train_dir = RUNS_DIR / TRAIN_NAME
    if train_dir.exists():
        print(f"输出目录已存在，为防止覆盖而停止：{train_dir}")
        print("如需重跑，请将脚本顶部 TRAIN_NAME 改为一个新名称。")
        return 1

    print("YOLOv8n：从 mosaic000 best.pt 开始进行 imgsz=800 短周期微调")
    print(f"初始权重：{BASE_MODEL}")
    print(f"数据配置：{DATA_YAML}")
    print(f"输出目录：{train_dir}")
    print(f"epochs={EPOCHS}, patience={PATIENCE}, batch={BATCH}, imgsz={IMGSZ}, mosaic=0.0")
    print("原有模型不会被覆盖。")

    try:
        model = YOLO(str(BASE_MODEL))
        model.train(
            task="detect",
            data=str(DATA_YAML),
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH,
            imgsz=IMGSZ,
            device=0,
            workers=0,
            project=str(RUNS_DIR),
            name=TRAIN_NAME,
            exist_ok=False,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=2.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.05,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=0,
            resume=False,
            amp=True,
            fraction=1.0,
            cache=False,
            multi_scale=0.0,
            val=True,
            split="val",
            save=True,
            save_period=-1,
            plots=True,
            iou=0.7,
            max_det=300,

            # Retain custom parameters used by the earlier project training runs.
            cls_remap=True,
            dis=6.0,
            box=7.5,
            cls=0.5,
            cls_pw=0.0,
            dfl=1.5,
            dlog=1.0,
            dgrad=0.5,
            dlam=1.0,
            nbs=64,

            # Keep the mosaic000 augmentation recipe, with Mosaic disabled.
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            bgr=0.0,
            mosaic=0.0,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            copy_paste_mode="flip",
            auto_augment="randaugment",
            erasing=0.4,
        )
    except Exception as exc:
        print("\n训练失败：")
        print(exc)
        traceback.print_exc()
        print("若是 CUDA out of memory，将 BATCH 从 4 改为 2，并把 TRAIN_NAME 改为新名称后重试。")
        print("若提示参数不受支持，请把完整报错发回，不要一次修改多个参数。")
        return 2

    new_model = train_dir / "weights" / "best.pt"
    if not new_model.is_file():
        print(f"未找到训练后的 best.pt：{new_model}")
        return 3

    try:
        validate(BASE_MODEL, 640, VAL_BASE_640)
        validate(BASE_MODEL, 800, VAL_BASE_800)
        validate(new_model, 800, VAL_FT_800)
    except Exception as exc:
        print("\n验证失败；已生成的训练权重会保留。错误如下：")
        print(exc)
        traceback.print_exc()
        return 4

    print("\n全部完成。请比较三个验证目录的六类 Precision、Recall、mAP50、mAP50-95。")
    print(f"微调权重：{new_model}")
    print(f"训练记录：{train_dir / 'results.csv'}")
    print(f"原模型 640 验证：{RUNS_DIR / VAL_BASE_640}")
    print(f"原模型 800 验证：{RUNS_DIR / VAL_BASE_800}")
    print(f"微调模型 800 验证：{RUNS_DIR / VAL_FT_800}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
