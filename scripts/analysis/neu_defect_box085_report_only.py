# -*- coding: utf-8 -*-
"""
只修复并生成 box=8.5 实验判定报告。
不会训练、不会运行模型推理、不会覆盖已有预测结果。
请在 E:\yolo_project 中运行。
"""
import csv
from pathlib import Path
import neu_defect_box085_postprocess_only as pipeline

runs = Path(r"E:\yolo_project\runs")
standard_path = runs / "neu_defect_box085_standard_validation_metrics.csv"
fixed_path = runs / "neu_defect_box085_fixed_threshold_overall.csv"
errors_path = runs / "neu_defect_box085_error_type_counts.csv"

for path in (standard_path, fixed_path, errors_path):
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少已生成的结果文件：{path}\n"
            "不要重新训练；请确认之前诊断推理生成的 CSV 是否在该目录。"
        )

def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

standard_rows = read_csv(standard_path)
fixed_rows = read_csv(fixed_path)
error_rows = read_csv(errors_path)

# CSV 读取结果默认是字符串，转换报告所需的数值字段，避免 f-string 格式化报错。
for row in standard_rows:
    for key in ("precision", "recall", "mAP50", "mAP50_95"):
        if row.get(key) not in (None, ""):
            row[key] = float(row[key])

for row in fixed_rows:
    for key in ("confidence_threshold", "match_iou", "micro_F1"):
        if row.get(key) not in (None, ""):
            row[key] = float(row[key])

for row in error_rows:
    for key in ("FP_localization_related", "FP_background",
                "FP_duplicate_or_competing", "FP_class_confusion_candidate"):
        if row.get(key) not in (None, ""):
            row[key] = int(float(row[key]))

pipeline.write_decision_report(standard_rows, fixed_rows, error_rows)

print("\n报告已生成，无需重新训练或重新推理：")
print(r"E:\yolo_project\runs\neu_defect_box085_decision_report.txt")
print("使用的是之前已生成的标准验证、固定阈值和错误类型 CSV。")
