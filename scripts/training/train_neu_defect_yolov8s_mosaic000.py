# -*- coding: utf-8 -*-
"""
NEU-DET 六类钢材表面缺陷：YOLOv8s 模型容量实验
不使用 Cursor Agent。

实验目的：
- 将 YOLOv8n 更换为更大的预训练 YOLOv8s，测试模型容量是否有助于复杂缺陷识别。
- 固定 mosaic=0.0，沿用前一轮表现较好的增强设置。
- 使用独立输出目录，不覆盖 neu_defect_clean、neu_defect_mosaic050 或 neu_defect_mosaic000。

已确认数据配置：
D:\\yolov8\\NEU-DET\\data.yaml

输出目录：
E:\\yolo_project\\runs\\neu_defect_yolov8s_mosaic000

说明：
- 这是一项新的模型规模对照实验，不是从 YOLOv8n 的 best.pt 继续训练。
- YOLOv8s 可能需要更多显存；本脚本默认 batch=8，适配 6GB 显存的保守起点。
- 如果提示 CUDA out of memory，将 BATCH 改成 4 后重试。
- 若 E:\\yolo_project\\yolov8s.pt 不存在，Ultralytics 会尝试获取官方 yolov8s.pt；
  如果当前环境无法联网，请先准备好该权重文件。
"""

from pathlib import Path
import sys

from ultralytics import YOLO

PROJECT_ROOT = Path(r"E:\yolo_project")
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
LOCAL_MODEL = PROJECT_ROOT / "yolov8s.pt"
RUNS_DIR = PROJECT_ROOT / "runs"
RUN_NAME = "neu_defect_yolov8s_mosaic000"

EPOCHS = 100
PATIENCE = 20
BATCH = 8
IMGSZ = 640


def main() -> int:
    if not PROJECT_ROOT.is_dir():
        print(f"找不到项目目录：{PROJECT_ROOT}")
        return 1
    if not DATA_YAML.is_file():
        print(f"找不到数据配置：{DATA_YAML}")
        return 1

    # 优先使用项目目录中的本地权重；没有时让 Ultralytics 尝试下载官方权重。
    model_source = str(LOCAL_MODEL) if LOCAL_MODEL.is_file() else "yolov8s.pt"

    print("=" * 78)
    print("NEU-DET 六类缺陷：YOLOv8s + Mosaic=0.0")
    print(f"初始权重：{model_source}")
    print(f"数据配置：{DATA_YAML}")
    print(f"输出目录：{RUNS_DIR / RUN_NAME}")
    print(f"epochs={EPOCHS}, patience={PATIENCE}, batch={BATCH}, imgsz={IMGSZ}")
    print("本轮主要变化：模型从 YOLOv8n 改为 YOLOv8s；Mosaic 固定为 0.0。")
    print("不会覆盖任何已有实验结果。")
    print("=" * 78)

    try:
        model = YOLO(model_source)
    except Exception as exc:
        print("\n加载 yolov8s.pt 失败。")
        print("请确认本地权重文件存在，或当前环境能够下载官方权重。")
        print(f"详细错误：{exc}")
        return 2

    train_args = dict(
        task="detect",
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        imgsz=IMGSZ,
        save=True,
        save_period=-1,
        cache=False,
        device="0",
        workers=0,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=False,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        multi_scale=0.0,
        compile=False,
        val=True,
        split="val",
        save_json=False,
        iou=0.7,
        max_det=300,
        plots=True,
        augment=False,
        agnostic_nms=False,

        # 沿用原 args.yaml 的优化器和学习率设置
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 沿用原训练配置中的损失参数
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

        # 沿用原训练配置中的增强设置，仅固定 Mosaic=0.0
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

    try:
        results = model.train(**train_args)
    except Exception as exc:
        print("\n训练启动或执行失败：")
        print(exc)
        print(
            "\n如果是显存不足，请将脚本顶部 BATCH 从 8 改为 4 后重试。"
            "\n如果提示参数不受支持，请把完整报错发回，不要同时修改多个参数。"
        )
        return 3

    save_dir = Path(getattr(results, "save_dir", RUNS_DIR / RUN_NAME))
    print("\n训练任务结束。")
    print(f"结果目录：{save_dir}")
    print(f"最佳权重：{save_dir / 'weights' / 'best.pt'}")
    print(f"训练记录：{save_dir / 'results.csv'}")
    print("\n训练完成后，先对 best.pt 做标准验证，再与 YOLOv8n 的 clean/mosaic000 比较六类指标。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
