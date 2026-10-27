# -*- coding: utf-8 -*-
"""
NEU-DET 六类缺陷：第一组可控训练实验
实验假设：原训练 mosaic=1.0 可能扰动纹理型缺陷的局部外观；
本实验只将 mosaic 从 1.0 改为 0.5，其余主要训练参数尽量沿用原 args.yaml。

注意：
- 从原始预训练 yolov8n.pt 重新训练，便于与原始 100 轮实验做相对公平的对照。
- 不会覆盖 runs/neu_defect_clean 或现有 best.pt。
- 这只是待验证的实验假设，不保证一定提升 crazing。
- 不使用 Cursor Agent。
"""

from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("找不到 ultralytics。请确认 Cursor 终端已激活原来的 yolo_workspace 环境。")
    raise

MODEL_PATH = Path(r"E:\yolo_project\yolov8n.pt")
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
PROJECT_DIR = Path(r"E:\yolo_project\runs")
RUN_NAME = "neu_defect_mosaic050"

def main() -> int:
    if not MODEL_PATH.is_file():
        print(f"找不到原始预训练权重：{MODEL_PATH}")
        print("请确认该文件存在；本脚本不会自动改用其他权重。")
        return 1
    if not DATA_YAML.is_file():
        print(f"找不到数据配置：{DATA_YAML}")
        return 1

    print("=" * 78)
    print("NEU-DET 六类缺陷训练实验：Mosaic 1.0 -> 0.5")
    print(f"初始化权重：{MODEL_PATH}")
    print(f"数据配置：{DATA_YAML}")
    print(f"新实验目录：{PROJECT_DIR / RUN_NAME}")
    print("本次唯一有意修改的主要训练参数：mosaic=0.5（原来为 1.0）")
    print("原来的 neu_defect_clean 和 best.pt 不会被覆盖。")
    print("=" * 78)

    model = YOLO(str(MODEL_PATH))

    # 参数依据用户提供的原 args.yaml。
    # 保留原来的训练设置，只将 mosaic 从 1.0 改为 0.5。
    train_args = dict(
        task="detect",
        data=str(DATA_YAML),
        epochs=100,
        patience=20,
        batch=16,
        imgsz=640,
        save=True,
        save_period=-1,
        cache=False,
        device="0",
        workers=0,
        project=str(PROJECT_DIR),
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
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # 原始检测损失相关设置（保留 args.yaml 中的数值）
        dis=6.0,
        box=7.5,
        cls=0.5,
        cls_pw=0.0,
        dfl=1.5,
        dlog=1.0,
        dgrad=0.5,
        dlam=1.0,
        nbs=64,
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
        mosaic=0.5,  # 本实验的主要改动：原值 1.0
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.4,
        cls_remap=True,
    )

    try:
        result = model.train(**train_args)
    except Exception as exc:
        print("\n训练启动或执行失败：")
        print(exc)
        print(
            "\n请把完整报错复制回来。由于原 args.yaml 含有 cls_remap、dis、"
            "dlog 等自定义参数，如果当前环境与原训练环境不同，可能需要对齐参数支持。"
        )
        return 2

    save_dir = getattr(result, "save_dir", PROJECT_DIR / RUN_NAME)
    print("\n训练任务结束。")
    print(f"结果目录：{save_dir}")
    print(f"优先检查：{Path(save_dir) / 'results.csv'}")
    print(f"最佳权重通常位于：{Path(save_dir) / 'weights' / 'best.pt'}")
    print("下一步使用同一份 data.yaml 验证新权重，并与 neu_defect_clean 的 best.pt 比较六类指标。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
