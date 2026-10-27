# -*- coding: utf-8 -*-
"""
NEU-DET：针对 crazing 的 1.5x 图像采样权重实验
Ultralytics 8.4.x / PyTorch

设计原则：
- 训练集仍然是原始 1440 张（不复制图片、不增加每个 epoch 的样本数）
- 对含 class 0（crazing）的图片，采样权重设为其他图片的 1.5 倍
- WeightedRandomSampler 每个 epoch 抽取 len(dataset) 张图片（有放回）
- 从原始预训练权重 yolov8n.pt 开始训练
- mosaic=0.0、cls=0.5、imgsz=640 等设置保持与 mosaic000 路线一致
- 训练前后用同一 data.yaml、同一验证参数评估 mosaic000 基线和新模型
- 不覆盖任何已有权重，不使用 Cursor Agent
"""

from pathlib import Path
import csv
import gc
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer

# ======================== 路径 ========================
PROJECT_ROOT = Path(r"E:\yolo_project")
RUNS_ROOT = PROJECT_ROOT / "runs"
DATA_YAML = Path(r"D:\yolov8\NEU-DET\data.yaml")
PRETRAINED_WEIGHT = PROJECT_ROOT / "yolov8n.pt"
BASELINE_WEIGHT = RUNS_ROOT / "neu_defect_mosaic000" / "weights" / "best.pt"

TRAIN_NAME = "neu_defect_crazing_weighted150"
BASELINE_VAL_NAME = "neu_defect_crazing_weighted150_baseline_fair_val"
CANDIDATE_VAL_NAME = "neu_defect_crazing_weighted150_candidate_fair_val"
SUMMARY_CSV = RUNS_ROOT / "neu_defect_crazing_weighted150_comparison.csv"
INFO_TXT = RUNS_ROOT / "neu_defect_crazing_weighted150_experiment.txt"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CRAZING_CLASS_ID = 0
CRAZING_IMAGE_WEIGHT = 1.5

# ======================== 训练设置 ========================
EPOCHS = 100
PATIENCE = 20
BATCH = 16
IMGSZ = 640
DEVICE = 0
WORKERS = 0
SEED = 42

# ======================== 统一验证设置 ========================
VAL_BATCH = 4
VAL_CONF = 0.001
VAL_IOU = 0.70
MAX_DET = 300


class CrazingWeightedTrainer(DetectionTrainer):
    """仅覆盖训练 DataLoader：以 1.5 倍权重抽取含 crazing 的图片。"""

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            return super().get_dataloader(
                dataset_path, batch_size=batch_size, rank=rank, mode=mode
            )

        # 本实验明确使用单 GPU；避免在 DDP 下重复使用同一 WeightedRandomSampler。
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise RuntimeError(
                "本脚本的加权采样实现用于单 GPU。请使用 device=0 运行，不要启用 DDP。"
            )

        dataset = self.build_dataset(dataset_path, mode, batch_size)
        if len(dataset) == 0:
            raise RuntimeError(f"训练数据集为空：{dataset_path}")
        if getattr(dataset, "rect", False):
            raise RuntimeError("加权采样实验要求 rect=False。")

        labels = getattr(dataset, "labels", None)
        if labels is None or len(labels) != len(dataset):
            raise RuntimeError(
                "无法从 Ultralytics 数据集中读取逐图片 labels，"
                "为避免错误采样，脚本停止。"
            )

        sample_weights = []
        crazing_images = 0
        images_with_other_classes_and_crazing = 0
        class_image_presence = np.zeros(len(CLASS_NAMES), dtype=np.int64)

        for label in labels:
            cls_values = np.asarray(label.get("cls", [])).reshape(-1)
            if cls_values.size:
                cls_ids = cls_values.astype(int)
            else:
                cls_ids = np.asarray([], dtype=int)

            present = set(int(x) for x in cls_ids if 0 <= int(x) < len(CLASS_NAMES))
            for cid in present:
                class_image_presence[cid] += 1

            has_crazing = CRAZING_CLASS_ID in present
            if has_crazing:
                crazing_images += 1
                if any(cid != CRAZING_CLASS_ID for cid in present):
                    images_with_other_classes_and_crazing += 1
                sample_weights.append(CRAZING_IMAGE_WEIGHT)
            else:
                sample_weights.append(1.0)

        if crazing_images == 0:
            raise RuntimeError(
                "训练标签中没有找到 class 0（crazing），请检查 data.yaml 和标签类别 ID。"
            )
        if crazing_images == len(dataset):
            raise RuntimeError(
                "所有训练图片都包含 crazing，1.5 倍加权将不会产生类别间差异。"
            )

        total_weight = float(sum(sample_weights))
        expected_crazing_draws = (
            len(dataset) * crazing_images * CRAZING_IMAGE_WEIGHT / total_weight
        )
        baseline_uniform_draws = len(dataset) * crazing_images / len(dataset)

        print("\n" + "-" * 72)
        print("Crazing 加权采样器已启用")
        print(f"训练图片总数：{len(dataset)}")
        print(f"包含 crazing 的图片数：{crazing_images}")
        print(f"含 crazing 且同时含其他类别的图片数：{images_with_other_classes_and_crazing}")
        print(f"crazing 图片采样权重：{CRAZING_IMAGE_WEIGHT:.2f}")
        print(f"每个 epoch 抽样数：{len(dataset)}（与原数据集长度相同）")
        print(
            "预计每个 epoch 抽到含 crazing 图片："
            f"{expected_crazing_draws:.1f} 张；均匀抽样约 {baseline_uniform_draws:.1f} 张"
        )
        print("各类至少出现于一张训练图片的图片数：")
        for cid, cname in enumerate(CLASS_NAMES):
            print(f"  {cname}: {int(class_image_presence[cid])}")
        print("-" * 72)

        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(SEED + 101)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=sampler_generator,
        )

        loader_generator = torch.Generator()
        loader_generator.manual_seed(SEED + 202)

        # 与 Ultralytics 的 InfiniteDataLoader 保持兼容；sampler 每轮重新抽样，
        # 但每轮样本数固定，不会因数据集扩容而额外增加每轮优化步数。
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=loader_generator,
            drop_last=bool(self.args.compile and mode == "train"),
        )


def check_paths_and_outputs():
    missing = [
        str(p)
        for p in [DATA_YAML, PRETRAINED_WEIGHT, BASELINE_WEIGHT]
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError("以下必需文件不存在：\n" + "\n".join(missing))

    if not RUNS_ROOT.is_dir():
        raise FileNotFoundError(f"运行目录不存在：{RUNS_ROOT}")

    planned = [
        RUNS_ROOT / TRAIN_NAME,
        RUNS_ROOT / BASELINE_VAL_NAME,
        RUNS_ROOT / CANDIDATE_VAL_NAME,
        SUMMARY_CSV,
        INFO_TXT,
    ]
    existing = [str(p) for p in planned if p.exists()]
    if existing:
        raise FileExistsError(
            "为避免覆盖已有结果，脚本停止。以下路径已存在：\n"
            + "\n".join(existing)
            + "\n如果确实需要重跑，请修改脚本顶部的 TRAIN_NAME、验证目录名和输出文件名。"
        )


def metric_rows(metrics, model_label):
    box = metrics.box
    class_ids = [int(x) for x in np.asarray(getattr(box, "ap_class_index", [])).reshape(-1)]
    p_values = np.asarray(getattr(box, "p", [])).reshape(-1)
    r_values = np.asarray(getattr(box, "r", [])).reshape(-1)
    ap50_values = np.asarray(getattr(box, "ap50", [])).reshape(-1)
    ap_values = np.asarray(getattr(box, "ap", [])).reshape(-1)

    rows = [{
        "model": model_label,
        "class_id": "all",
        "class_name": "all",
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
        "mAP50": round(float(box.map50), 6),
        "mAP50_95": round(float(box.map), 6),
    }]

    for cid, cname in enumerate(CLASS_NAMES):
        if cid in class_ids:
            idx = class_ids.index(cid)
            p = float(p_values[idx]) if idx < len(p_values) else 0.0
            r = float(r_values[idx]) if idx < len(r_values) else 0.0
            ap50 = float(ap50_values[idx]) if idx < len(ap50_values) else 0.0
            ap = float(ap_values[idx]) if idx < len(ap_values) else 0.0
        else:
            p = r = ap50 = ap = 0.0
        rows.append({
            "model": model_label,
            "class_id": cid,
            "class_name": cname,
            "precision": round(p, 6),
            "recall": round(r, 6),
            "mAP50": round(ap50, 6),
            "mAP50_95": round(ap, 6),
        })
    return rows


def save_comparison(rows):
    fields = [
        "model", "class_id", "class_name",
        "precision", "recall", "mAP50", "mAP50_95",
    ]
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_model(weight_path, run_name, label):
    print("\n" + "=" * 76)
    print(f"统一验证：{label}")
    print(f"权重：{weight_path}")
    print(
        f"imgsz={IMGSZ}, batch={VAL_BATCH}, conf={VAL_CONF}, "
        f"iou={VAL_IOU}, max_det={MAX_DET}"
    )
    print("=" * 76)

    model = YOLO(str(weight_path))
    metrics = model.val(
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
    rows = metric_rows(metrics, label)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    check_paths_and_outputs()

    print("=" * 76)
    print("NEU-DET：crazing 定向加权采样实验")
    print("训练起点：", PRETRAINED_WEIGHT)
    print("对照基线：", BASELINE_WEIGHT)
    print("数据配置：", DATA_YAML)
    print("训练输出：", RUNS_ROOT / TRAIN_NAME)
    print("采样策略：含 crazing 图片权重 1.5，其余图片权重 1.0")
    print("训练设置：epochs=100, patience=20, batch=16, imgsz=640, mosaic=0.0, cls=0.5")
    print("每个 epoch 抽样数保持原训练图片数，不增加每轮训练步数。")
    print("=" * 76)

    all_rows = []

    # 1. 先评估 mosaic000 基线，建立本次严格同口径的比较值。
    baseline_rows = validate_model(
        BASELINE_WEIGHT,
        BASELINE_VAL_NAME,
        "neu_defect_mosaic000_baseline",
    )
    all_rows.extend(baseline_rows)
    save_comparison(all_rows)

    # 2. 从与 mosaic000 相同的原始预训练权重开始训练，只改变训练图片采样权重。
    print("\n开始训练：仅改变训练图片的抽样权重。")
    model = YOLO(str(PRETRAINED_WEIGHT))
    model.train(
        trainer=CrazingWeightedTrainer,
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
        cls=0.5,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_weight = RUNS_ROOT / TRAIN_NAME / "weights" / "best.pt"
    if not best_weight.is_file():
        raise FileNotFoundError(
            "训练结束后没有找到最佳权重：\n" + str(best_weight)
        )

    # 3. 对新权重使用与基线相同的验证配置。
    candidate_rows = validate_model(
        best_weight,
        CANDIDATE_VAL_NAME,
        "neu_defect_crazing_weighted150",
    )
    all_rows.extend(candidate_rows)
    save_comparison(all_rows)

    info = [
        "NEU-DET crazing weighted-sampling experiment",
        f"data_yaml={DATA_YAML}",
        f"pretrained_weight={PRETRAINED_WEIGHT}",
        f"baseline_weight={BASELINE_WEIGHT}",
        "sampling_weight_crazing_image=1.5",
        "sampling_weight_other_image=1.0",
        "WeightedRandomSampler replacement=True",
        "samples_per_epoch=len(training_dataset)",
        f"imgsz={IMGSZ}",
        f"batch={BATCH}",
        f"epochs_max={EPOCHS}",
        f"patience={PATIENCE}",
        "mosaic=0.0",
        "cls=0.5",
        f"seed={SEED}",
        f"validation_conf={VAL_CONF}",
        f"validation_iou={VAL_IOU}",
        "Note: weighting is image-level. If an image contains both crazing and another class, that image is weighted 1.5x as a whole.",
        "Train-set metrics are not independent test results; all model selection here uses the same validation split.",
        f"best_weight={best_weight}",
        f"comparison_csv={SUMMARY_CSV}",
    ]
    INFO_TXT.write_text("\n".join(info), encoding="utf-8")

    print("\n" + "=" * 76)
    print("定向采样实验完成。")
    print("训练记录：", RUNS_ROOT / TRAIN_NAME / "results.csv")
    print("新权重：", best_weight)
    print("基线验证：", RUNS_ROOT / BASELINE_VAL_NAME)
    print("新模型验证：", RUNS_ROOT / CANDIDATE_VAL_NAME)
    print("统一指标对比：", SUMMARY_CSV)
    print("实验设置记录：", INFO_TXT)
    print("=" * 76)
    print("请以六类 mAP50、mAP50-95、crazing Recall，以及 rolled-in_scale 的 Precision/Recall 综合决定是否保留新模型。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n脚本运行失败：")
        print(exc)
        traceback.print_exc()
        sys.exit(1)
