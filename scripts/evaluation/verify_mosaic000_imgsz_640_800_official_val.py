# -*- coding: utf-8 -*-
"""
用 Ultralytics 官方 model.val() 复核同一份 mosaic000 权重在 imgsz=640 与 800 时的表现。
只做验证，不训练，不改权重。除 imgsz 外，其他显式评估参数保持一致。

若自动检测不到唯一的数据 YAML，请在 DATA_YAML 中填写实际 YAML 路径。
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ===================== 路径配置 =====================
WEIGHT = Path(r"E:\yolo_project\runs\neu_defect_mosaic000\weights\best.pt")
DATASET_ROOT = Path(r"D:\yolov8\NEU-DET")
PROJECT_ROOT = Path(r"E:\yolo_project\runs\neu_defect_imgsz_official_review")

# 如果自动发现 YAML 不唯一或失败，请在此处填写实际数据 YAML，例如：
# DATA_YAML = Path(r"E:\yolo_project\your_dataset.yaml")
DATA_YAML = None

# ===================== 固定验证配置 =====================
IMGSZ_VALUES = (640, 800)
BATCH = 2                    # 两次验证使用相同 batch，降低 6 GB 显存压力
CONF = 0.001                 # Ultralytics 检测验证常用低置信度阈值
NMS_IOU = 0.70               # 注意：这是验证时的 NMS IoU，不是前一阶段的 0.30
MAX_DET = 300
WORKERS = 0
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SPLIT = "val"
RECT = True
HALF = False
AUGMENT = False
PLOTS = True


def resolve_dataset_yaml() -> Path:
    if DATA_YAML is not None:
        p = Path(DATA_YAML)
        if not p.is_file():
            raise FileNotFoundError(f"DATA_YAML 指定的文件不存在：{p}")
        return p

    search_roots = [
        DATASET_ROOT,
        Path(r"E:\yolo_project"),
    ]
    candidates = []
    seen = set()

    expected_val = (DATASET_ROOT / "validation" / "images").resolve()

    for root in search_roots:
        if not root.exists():
            continue
        for p in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
            if not p.is_file() or p.resolve() in seen:
                continue
            seen.add(p.resolve())
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
                if not isinstance(data, dict) or "val" not in data:
                    continue

                dataset_path = data.get("path", "")
                if dataset_path:
                    dp = Path(str(dataset_path))
                    if not dp.is_absolute():
                        dp = (p.parent / dp).resolve()
                else:
                    dp = p.parent.resolve()

                val_value = Path(str(data["val"]))
                if val_value.is_absolute():
                    val_path = val_value.resolve()
                else:
                    val_path = (dp / val_value).resolve()

                names = data.get("names")
                nc = data.get("nc")
                class_count_ok = (
                    (isinstance(names, (list, tuple)) and len(names) == 6)
                    or (isinstance(names, dict) and len(names) == 6)
                    or nc == 6
                )
                if val_path == expected_val and class_count_ok:
                    candidates.append(p.resolve())
            except Exception:
                # 忽略不相关或无法解析的 YAML，继续寻找
                continue

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "没有自动找到唯一匹配 NEU-DET 的数据 YAML。\n"
            f"请编辑脚本，将 DATA_YAML 设置为实际 YAML 路径。\n"
            f"预期验证图片目录：{expected_val}"
        )

    candidate_text = "\n".join(str(p) for p in candidates)
    raise RuntimeError(
        "找到多个匹配的数据 YAML，避免误用数据配置，脚本已停止。\n"
        "请编辑脚本，将 DATA_YAML 设置为其中正确的一个：\n"
        f"{candidate_text}"
    )


def write_summary(rows: list[dict], out_path: Path) -> None:
    if not rows:
        raise RuntimeError("没有验证指标，不能生成汇总 CSV。")
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not WEIGHT.is_file():
        raise FileNotFoundError(f"模型权重不存在：{WEIGHT}")
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在：{DATASET_ROOT}")
    if PROJECT_ROOT.exists():
        for size in IMGSZ_VALUES:
            run_dir = PROJECT_ROOT / f"imgsz_{size}"
            if run_dir.exists():
                raise FileExistsError(
                    f"输出目录已存在，为避免覆盖旧结果，脚本停止：{run_dir}\n"
                    "请更改 PROJECT_ROOT，或先归档旧输出后再运行。"
                )

    data_yaml = resolve_dataset_yaml()
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("NEU-DET 官方验证复核：同一权重，只改变 imgsz", flush=True)
    print(f"权重：{WEIGHT}", flush=True)
    print(f"数据 YAML：{data_yaml}", flush=True)
    print(f"验证图片目录预期：{DATASET_ROOT / 'validation' / 'images'}", flush=True)
    print(
        f"固定参数：split={SPLIT}, batch={BATCH}, conf={CONF}, "
        f"NMS IoU={NMS_IOU}, max_det={MAX_DET}, rect={RECT}, "
        f"half={HALF}, augment={AUGMENT}, device={DEVICE}, workers={WORKERS}",
        flush=True,
    )
    print("注意：本脚本不调用 train()，不会启动训练。", flush=True)
    print("=" * 78, flush=True)

    model = YOLO(str(WEIGHT))
    rows = []

    for imgsz in IMGSZ_VALUES:
        print("\n" + "=" * 78, flush=True)
        print(f"开始官方 model.val()：imgsz={imgsz}", flush=True)
        print("=" * 78, flush=True)

        metrics = model.val(
            data=str(data_yaml),
            split=SPLIT,
            imgsz=imgsz,
            batch=BATCH,
            conf=CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            device=DEVICE,
            workers=WORKERS,
            rect=RECT,
            half=HALF,
            augment=AUGMENT,
            plots=PLOTS,
            save_json=False,
            verbose=True,
            project=str(PROJECT_ROOT),
            name=f"imgsz_{imgsz}",
            exist_ok=False,
        )

        results = getattr(metrics, "results_dict", {}) or {}
        row = {
            "weight": str(WEIGHT),
            "data_yaml": str(data_yaml),
            "imgsz": imgsz,
            "split": SPLIT,
            "batch": BATCH,
            "conf": CONF,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
            "rect": RECT,
            "half": HALF,
            "augment": AUGMENT,
            "device": DEVICE,
        }
        for key, value in results.items():
            try:
                row[str(key)] = float(value)
            except (TypeError, ValueError):
                row[str(key)] = str(value)

        rows.append(row)
        print(f"\nimgsz={imgsz} 汇总指标：", flush=True)
        for key, value in results.items():
            print(f"  {key}: {value}", flush=True)

    summary_path = PROJECT_ROOT / "official_val_640_vs_800_summary.csv"
    write_summary(rows, summary_path)

    print("\n" + "=" * 78, flush=True)
    print("两次官方验证已完成。", flush=True)
    print(f"汇总 CSV：{summary_path}", flush=True)
    print("每次验证的类别指标、曲线及混淆矩阵（若 Ultralytics 生成）保存在：", flush=True)
    print(f"  {PROJECT_ROOT / 'imgsz_640'}", flush=True)
    print(f"  {PROJECT_ROOT / 'imgsz_800'}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
