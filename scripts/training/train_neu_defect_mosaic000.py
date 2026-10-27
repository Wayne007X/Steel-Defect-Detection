
from pathlib import Path
from ultralytics import YOLO


def main():
    model_path = Path(r"E:\yolo_project\yolov8n.pt")
    data_path = Path(r"D:\yolov8\NEU-DET\data.yaml")
    project_path = Path(r"E:\yolo_project\runs")

    for path in (model_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(f"找不到文件：{path}")

    print("=" * 65)
    print("YOLOv8 六类钢材缺陷检测：第三轮 Mosaic 消融实验")
    print(f"初始模型：{model_path}")
    print(f"数据配置：{data_path}")
    print("Mosaic：0.0")
    print("其他主要训练参数与 clean 基线保持一致")
    print("=" * 65)

    model = YOLO(str(model_path))

    results = model.train(
        data=str(data_path),
        epochs=100,
        patience=20,
        batch=16,
        imgsz=640,

        device=0,
        workers=0,

        project=str(project_path),
        name="neu_defect_mosaic000",
        exist_ok=False,

        pretrained=True,
        cls_remap=True,
        optimizer="auto",
        seed=42,
        deterministic=True,

        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        multi_scale=0.0,

        val=True,
        split="val",
        iou=0.7,
        max_det=300,
        plots=True,
        save=True,
        save_period=-1,
        cache=False,

        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        dis=6.0,
        box=7.5,
        cls=0.5,
        cls_pw=0.0,
        dfl=1.5,
        dgrad=0.5,
        dlam=1.0,
        dlog=1.0,
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

        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",

        auto_augment="randaugment",
        erasing=0.4,
    )

    print("\n训练结束")
    print(f"结果目录：{results.save_dir}")
    print(f"最佳权重：{Path(results.save_dir) / 'weights' / 'best.pt'}")
    print(f"训练记录：{Path(results.save_dir) / 'results.csv'}")


if __name__ == "__main__":
    main()
