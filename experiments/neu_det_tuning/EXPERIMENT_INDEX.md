# NEU-DET 实验索引与结论

> 本文依据当前归档包中的训练脚本、`args.yaml`、训练 `results.csv`、标准验证 CSV、固定阈值 CSV、错误分析结果及正式报告整理。
>
> **重要边界：** 当前项目没有独立测试集。下述结论均是现有 NEU-DET 验证集上的实验结论，不等于独立测试或真实产线验收结果。

## 1. 项目与评估口径

- 数据集：NEU-DET
- 训练集：1,440 张图像
- 验证集：360 张图像、854 个真实目标
- 类别：`crazing`、`inclusion`、`patches`、`pitted_surface`、`rolled-in_scale`、`scratches`
- 当前主线输入尺寸：640
- 主要训练环境记录：Python 3.10.11、PyTorch 2.5.1+cu121、Ultralytics 8.4.177
- 主要候选：`neu_defect_clean`、`neu_defect_mosaic000`

### 指标不要混用

- **标准验证 P/R、mAP**：来自 Ultralytics 官方验证流程；P/R 是其 PR 曲线工作点结果。
- **固定阈值 P/R/F1**：在指定 confidence、NMS IoU 和预测框—真实框匹配 IoU 下统计，代表特定工作点。
- **NMS IoU** 与 **匹配 IoU** 是两个不同参数。
- 不同来源、不同阈值、不同最佳 epoch 的指标不能拼在同一行当作同一次评估结果。

## 2. 核心实验总表

| 实验组 | 核心问题 / 改动 | 主要证据 | 结论与状态 |
|---|---|---|---|
| `neu_defect_train` | 早期历史模型；seed=0、最多 50 epoch、mosaic=1.0 等。训练时数据集尚未纠正：241 张训练图片对应的标签/XML 被放到了验证集，训练图像与标注的位置不匹配。 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_train/args.yaml`、`results.csv`；`deployment_model_comparison/three_model_standard_metrics.csv`；用户对当时数据集问题的说明 | **数据集纠正前的历史模型，不可作为当前主基线，也不能与 clean/mosaic000 视为公平对照。** 三模型对比 CSV 中该模型的 `pitted_surface` 和 `rolled-in_scale` 指标为 0。已知的数据集错位是重要混杂因素，因此这些零指标不能直接解释为模型本身不会检测对应类别。 |
| `neu_defect_clean` | Mosaic=1.0 的主线基线 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_clean/args.yaml`、`results.csv`、`experiments/neu_det_tuning/results/raw_runs/neu_defect_tuning_stage123_no_training_20261026/standard_validation_metrics.csv` | **整体 mAP 基线。** 标准 mAP@0.5=0.7281，mAP@0.5:0.95=0.3844。 |
| `neu_defect_mosaic050` | 从 clean 路线将 mosaic 从 1.0 改为 0.5 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_mosaic050/args.yaml`、`results.csv`、标准验证 CSV | **已完成；不如 mosaic000 的总体 Recall/mAP@0.5。** 标准 P=0.6456、R=0.6685、mAP@0.5=0.7111、mAP@0.5:0.95=0.3781。 |
| `neu_defect_mosaic000` | 从 clean 路线将 mosaic 从 1.0 改为 0.0 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_mosaic000/args.yaml`、`results.csv`、标准验证 CSV | **召回优先候选。** 标准 P=0.6639、R=0.6886、mAP@0.5=0.7260、mAP@0.5:0.95=0.3784。相对 clean，Recall 更高但 mAP 略低。 |
| `neu_defect_yolov8s_mosaic000` | 将模型规模由 YOLOv8n 改为 YOLOv8s，mosaic=0 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_yolov8s_mosaic000/args.yaml`、`results.csv`、标准验证 CSV | **未取代 YOLOv8n。** 标准 mAP@0.5=0.7078、mAP@0.5:0.95=0.3662，整体 Recall=0.6515；本次条件下没有体现更大模型的收益。 |
| `neu_defect_mosaic000_img800_ft` | 从 mosaic000 权重出发，以 800 输入进行 30 epoch 微调 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_mosaic000_img800_ft/args.yaml`、`results.csv`、标准验证 CSV | **未取代 640 主线。** 微调同时改变了输入尺寸、batch、optimizer、学习率、warmup 和 close_mosaic 等设置，不是严格的“只改变 imgsz”实验。另有同一权重的官方 640/800 复核，见第 3 节。 |
| `neu_defect_mosaic000_cls075_controlled` | 从 mosaic000 `best.pt` 开始微调，只把 cls 从 0.5 改为 0.75 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_mosaic000_cls075_controlled/args.yaml`、`results.csv`、`experiments/neu_det_tuning/results/raw_runs/neu_defect_mosaic000_cls075_comparison.csv` | **未通过整体改善判断。** 标准 mAP@0.5=0.7243、mAP@0.5:0.95=0.3720，低于 mosaic000；整体 Recall 也更低。部分类别 mAP 有涨有跌。训练记录共 21 epoch，最佳 mAP@0.5:0.95 出现在第 1 轮。 |
| `neu_defect_mosaic000_cls075`（无 controlled 后缀） | 另一个 cls=0.75 设计；脚本声明从原始 `yolov8n.pt` 开始独立训练 | `scripts/training/train_neu_defect_mosaic000_cls075.py` | **与 controlled 版本不是同一实验。** 当前归档包没有与该脚本对应的专属 `args.yaml`/结果文件；不要把它和 controlled 结果混为一谈，也不要仅凭文件存在就认为该方案已完成验证。 |
| `neu_defect_crazing_weighted150` | 对含 crazing 的训练图像使用 1.5 倍采样权重；每个 epoch 抽样数仍为数据集长度 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_crazing_weighted150/args.yaml`、`results.csv`、`neu_defect_crazing_weighted150_comparison.csv`、`neu_defect_crazing_weighted150_experiment.txt` | **未达到预期。** 整体 mAP@0.5=0.7162、mAP@0.5:0.95=0.3702；crazing mAP@0.5 约 0.4708、Recall 约 0.3210，均低于 mosaic000 对应结果。 |
| `neu_defect_box085_controlled` | 从原始预训练 YOLOv8n 开始，mosaic=0，box loss weight 从 7.5 改为 8.5 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_box085_controlled/args.yaml`、`results.csv`、`neu_defect_box085_standard_validation_metrics.csv`、`neu_defect_box085_decision_report.txt` | **明确未通过预设验收门槛。** mAP@0.5=0.7182、mAP@0.5:0.95=0.3705；crazing mAP@0.5 明显下降；多个验收门槛失败。不替换现有候选。 |
| `neu_defect_mosaic000_lr0005` | 目标初始学习率约 0.0005 的训练尝试 | `experiments/neu_det_tuning/results/raw_runs/neu_defect_mosaic000_lr0005/args.yaml`、`results.csv`、正式报告 | **未证明有足够收益。** `args.yaml` 同时显示 optimizer 从基线的 `auto` 改成 `AdamW`，因此不能把变化归因于学习率单一因素。最佳 mAP@0.5 与最佳 mAP@0.5:0.95 出现在不同 epoch；需避免拼接为同一检查点的指标。 |
| 同权重 `imgsz=640` vs `800` 官方复核 | 固定 mosaic000 同一 `best.pt`，只改变验证输入尺寸 | `experiments/neu_det_tuning/results/logs/imgsz_640_800_console_log.txt`、`scripts/evaluation/verify_mosaic000_imgsz_640_800_official_val.py` | **当前支持 640。** 640：mAP@0.5=0.725950、mAP@0.5:0.95=0.378362；800：0.639813、0.278748。该结论只适用于当前权重、数据与评估配置。 |
| clean vs mosaic000 固定阈值比较 | 相同推理设置下比较部署工作点 | `experiments/neu_det_tuning/results/raw_runs/clean_mosaic_fixed_threshold_compare_20261026/fixed_threshold_overall.csv`、`comparison_summary.txt` | **召回优先选 mosaic000；误报成本更低时 clean 也有价值。** 详细数值见第 3 节。 |
| `scratches_postprocess` 第一轮 | 划痕类别自定义重复框过滤；细长框垂直中心距离参数为 0.25 | `scripts/postprocessing/scratches_postprocess.py`、`experiments/neu_det_tuning/results/raw_runs/detect/scratches_evaluation/overall_metrics.csv` | **在当前 60 张划痕验证子集上有收益。** match IoU=0.50 时，F1 从 0.5756 升到 0.7021，FP 从 121 降至 59，TP/FN 不变。不是独立测试结论。 |
| `scratches_postprocess_v2` 第二轮 | 将上述细长框参数从 0.25 改为 0.20 | `scripts/postprocessing/scratches_postprocess_v2.py`、划痕评估 CSV | **略逊于第一轮。** match IoU=0.50 时 F1=0.6947、FP=62；没有证据支持用 v2 替换第一轮。 |
| 划痕置信度扫描 | 在同一批候选预测上扫描 conf=0.05–0.70，对比 baseline、filter_0.25、filter_0.20 | `scripts/postprocessing/sweep_scratches_confidence.py`、`experiments/neu_det_tuning/results/raw_runs/detect/scratches_conf_sweep/confidence_sweep_metrics.csv` | **仅作为当前验证子集上的阈值探索。** 由于阈值是在同一验证子集上选择，最佳阈值不应直接当成独立泛化结果。 |
| 标签完整性审计 | 检查 train/validation 标签文件和边界框 | `scripts/analysis/audit_neu_defect_labels.py`、`experiments/neu_det_tuning/results/raw_runs/detect/label_audit/` | **归档结果显示完整性检查通过。** train 1440 张、validation 360 张；缺失标签、无对应图片标签、空标签文件、超出归一化范围框的计数均为 0。 |
| train/val 错误分析 | 统计 GT、TP、FP、FN 和错误类型，定位弱类别 | `scripts/analysis/neu_defect_train_val_error_comparison_v2.py`、`experiments/neu_det_tuning/results/raw_runs/neu_defect_train_val_error_comparison_v2/` | **诊断证据，不是训练改进实验。** 可用于提出后续假设；错误类型归因受匹配规则与置信度阈值影响，不能单独证明根因。 |
| 阶段 1/2/3 无训练复核 | 对现有权重做统一验证、阈值网格比较、汇总已有错误分析 | `scripts/analysis/neu_defect_stage123_no_training_ready.py`、`experiments/neu_det_tuning/results/raw_runs/neu_defect_tuning_stage123_no_training_20261026/` | **评估流程，不会训练。** 用于收束候选、避免为了单一指标继续追加训练。 |

## 3. 当前最重要的模型对照

### 3.1 标准验证结果

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 当前定位 |
|---|---:|---:|---:|---:|---|
| `neu_defect_clean` | 0.6731 | 0.6495 | **0.7281** | **0.3844** | 整体 mAP 基线 |
| `neu_defect_mosaic000` | 0.6639 | **0.6886** | 0.7260 | 0.3784 | 召回优先候选 |
| `neu_defect_mosaic050` | 0.6456 | 0.6685 | 0.7111 | 0.3781 | Mosaic 中间方案 |
| `neu_defect_yolov8s_mosaic000` | 0.6635 | 0.6515 | 0.7078 | 0.3662 | 更大模型未体现收益 |
| `neu_defect_mosaic000_cls075_controlled` | 0.6748 | 0.6768 | 0.7243 | 0.3720 | 未替换基线 |
| `neu_defect_crazing_weighted150` | 0.6255 | 0.6880 | 0.7162 | 0.3702 | 未达到预期 |
| `neu_defect_box085` | 0.7090 | 0.6204 | 0.7182 | 0.3705 | 未通过验收 |

这些指标来自统一标准验证归档；不要与固定阈值工作点表混用。

### 3.2 clean vs mosaic000：固定工作点

条件：imgsz=640、NMS IoU=0.30、匹配 IoU=0.50、验证集 360 张、GT=854。

| 模型 | conf | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.20 | 0.7002 | 0.6838 | 0.6919 | 584 | 250 | 270 |
| mosaic000 | 0.20 | 0.6905 | **0.7002** | 0.6953 | 598 | 268 | 256 |
| clean | 0.25 | 0.7391 | 0.6534 | 0.6936 | 558 | 197 | 296 |
| mosaic000 | 0.25 | 0.7315 | 0.6698 | **0.6993** | 572 | 210 | 282 |

解释：

- 若当前目标是降低漏检，mosaic000 在这两个工作点下 TP 更多、FN 更少，但 FP 也更多。
- 若更重视控制误报，clean 的 FP 较少。
- `conf=0.20 / NMS IoU=0.30` 是召回优先的试运行起点；`conf=0.25 / NMS IoU=0.30` 是当前验证集上的 F1 优先候选。
- 这些阈值是在现有验证集上比较出来的，投入实际使用前仍需独立数据和业务成本验证。

## 4. 划痕后处理的结果

评估范围为 60 张划痕验证图片、124 个真实目标。固定 conf=0.30；匹配 IoU 是预测框与真实框的一对一匹配门槛，不是 NMS IoU。

| 方案 | 匹配 IoU | 预测数 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 内置 NMS 后、未自定义过滤 | 0.50 | 220 | 99 | 121 | 25 | 0.4500 | 0.7984 | 0.5756 |
| 第一轮 filter 0.25 | 0.50 | 158 | 99 | 59 | 25 | 0.6266 | 0.7984 | **0.7021** |
| 第二轮 filter 0.20 | 0.50 | 161 | 99 | 62 | 25 | 0.6149 | 0.7984 | 0.6947 |
| 内置 NMS 后、未自定义过滤 | 0.75 | 220 | 58 | 162 | 66 | 0.2636 | 0.4677 | 0.3372 |
| 第一轮 filter 0.25 | 0.75 | 158 | 57 | 101 | 67 | 0.3608 | 0.4597 | **0.4043** |
| 第二轮 filter 0.20 | 0.75 | 161 | 57 | 104 | 67 | 0.3540 | 0.4597 | 0.4000 |

第一轮过滤在当前子集的固定工作点上减少了大量 FP，且 IoU=0.50 时 TP/FN 不变；但这不是“已证明不会误删真实划痕”，也不是独立测试结果。需要保留可视化与逐框明细用于人工抽查。

## 5. 容易混淆的脚本与归档问题

### 5.1 同一实验有多个脚本名

- `scripts/training/train_neu_defect_box085_controlled.py`
- `scripts/training/train_neu_defect_box085_controlled_fixed_paths.py`
- `scripts/postprocessing/neu_defect_box085_postprocess_only.py`

这三个文件都属于 **box=8.5 训练/验证流程**，并非三个不同实验。它们不是完全相同的字节内容，但使用相同的核心实验名称/输出命名。不要连续运行它们来“再做三次实验”；这会导致输出冲突或重复训练。建议保留一个明确的 canonical（主版本）脚本，其他版本在 README 中标为旧版/修正版，并写明差异。

### 5.2 cls=0.75 有两个不同设计

- `train_neu_defect_mosaic000_cls075.py`：从原始预训练权重开始训练。
- `train_neu_defect_mosaic000_cls075_controlled.py`：从 `neu_defect_mosaic000/weights/best.pt` 开始微调。

由于训练起点不同，这两个结果不能互相替代。当前归档的 `neu_defect_mosaic000_cls075_comparison.csv` 对应 controlled 版本；未发现无 controlled 版本对应的完整专属结果归档。

### 5.3 “学习率实验”不只是学习率变化

`neu_defect_mosaic000_lr0005/args.yaml` 中 `optimizer=AdamW`，而 `neu_defect_mosaic000/args.yaml` 中 `optimizer=auto`。因此当前实验更准确的描述是“AdamW + lr0≈0.0005 方案对照”，不是纯粹的学习率单变量实验。

### 5.4 `neu_defect_train` 使用了数据集纠正前的错误划分

根据项目负责人对当时数据集整理过程的说明：

- 训练 `neu_defect_train` 时，数据集尚未纠正；有 241 张训练图片对应的标签和 XML 文件被放到了验证集，造成图像与标注位置不匹配。
- 后续数据集位置经过核对与纠正。
- `neu_defect_clean` 和 `neu_defect_mosaic000` 均使用纠正后的数据集训练。

因此，`neu_defect_train` 与后两个模型的训练数据条件不同。`deployment_model_comparison/three_model_standard_metrics.csv` 中该旧模型的 `pitted_surface` 和 `rolled-in_scale` 指标为 0，不能直接解释为模型本身无法检测这两类；已知的数据集错位是重要混杂因素。

**处理原则：** 保留 `neu_defect_train` 的脚本、权重来源记录和历史结果作为项目过程证据，但将其标记为“数据集纠正前的历史模型”，不用于当前主线模型的公平性能比较，也不作为当前六类检测基线。当前主线基线为 `neu_defect_clean`（整体 mAP 参考）和 `neu_defect_mosaic000`（召回优先候选）。

## 6. 建议的规范状态标签

后续所有实验在索引中只使用以下状态，避免“做过”与“有效”混为一谈：

- **主基线**：用于对照，参数与权重来源明确。
- **候选模型**：结果值得进一步验证，但尚未通过独立业务验证。
- **已完成 / 未通过**：实验运行完成，但未达到预设验收门槛。
- **诊断工具**：用于验证、审计、错误分析，不代表训练改进。
- **探索性实验**：参数或对照条件不够纯，结论需要保守解释。
- **旧版 / 重复脚本**：保留历史证据，但不作为推荐入口。
- **待核查**：数据或模型输出之间存在未解释的差异。

## 7. 当前结论

1. `neu_defect_clean` 保留为整体 mAP 基线。
2. `neu_defect_mosaic000` 保留为召回优先候选。
3. 当前没有证据支持将 cls=0.75、crazing weighted sampling、box=8.5、lr0005 或 YOLOv8s 作为新的主模型。
4. 当前输入尺寸主线保留 640；同权重官方复核不支持把输入尺寸提高到 800。
5. 划痕后处理第一轮在现有 60 张验证子集的固定工作点下优于第二轮，但还需要独立数据和人工审查。
6. 公开仓库中的训练脚本应标明本地数据/权重路径要求；归档指标可以公开，但不要暗示读者克隆仓库后就能直接重现训练，因为数据集和模型权重未包含在仓库中。
7. 下一阶段优先做候选模型的独立数据或业务验证，不要因为某个类别的单项指标不理想而无计划地继续追加训练。
