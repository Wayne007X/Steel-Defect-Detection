# 钢铁缺陷检测精度升级包（2026-10-30）

## 推荐使用方式

本轮已完成新权重微调、原模型对照、模型融合、TTA、全验证集推理复核及本地 API 测试。
所有结果来自现有 NEU-DET 验证集 360 张图片、854 个目标；训练集为 1440 张。
原项目的权重、数据、接口源文件和现有服务没有被替换。

### 1. 优先精度：GPU 双模型融合 + TTA

使用原 clean 权重与本包新 texture_ft 权重，推荐 conf=0.25、NMS IoU=0.30。

- 实际可调用推理实现的 mAP@0.5：76.7061%。
- 同一实现 mAP@0.5:0.95：38.3696%。
- 固定 conf=0.25：Precision=74.5783%，Recall=72.4824%，F1=73.5154%。
- TP=619、FP=211、FN=235。
- RTX 3060 Laptop 6GB、batch=1、训练结束后实测：单张中位耗时 37.11 ms，p95 44.69 ms；包含读图、预处理、两个模型、TTA、NMS 和融合，不含绘图保存与网页网络耗时。

在本目录运行（将图片路径换成你的图片）：

```powershell
& 'E:\yolo_workspace\Scripts\python.exe' .\predict_accuracy.py 'D:\yolov8\NEU-DET\validation\images\crazing\crazing_241.jpg' --tta --device cuda:0 --conf 0.25 --iou 0.30
```

结果图片和 JSON 写入当前目录的 accuracy_predictions。
如果更看重边界框定位，在同一条命令中改用 --iou 0.70；对应 mAP@0.5=76.0684%、mAP@0.5:0.95=40.4840%。
这两组指标属于不同配置，不能拼成一组结果。

### 2. 接入现有网页：CPU ONNX 双模型融合

CPU ONNX 模式不执行 TTA，使用 clean ONNX 与新 texture_ft ONNX。
全量实际推理 mAP@0.5=75.4612%、mAP@0.5:0.95=39.7547%，两者对应 NMS IoU=0.70。
单张中位耗时 104.78 ms，p95 113.67 ms，batch=1、每个 ORT session 使用 2 个 CPU 线程。
这是 CPU 的测量值，不能与 GPU 模式直接换算加速比。

终端一进入本目录后执行：

```powershell
.\run_accuracy_api.ps1
```

终端二进入本目录后执行：

```powershell
.\run_accuracy_frontend.ps1
```

打开 http://127.0.0.1:8502 。侧栏后端应为 http://127.0.0.1:8001 。
新端口保留现有 8000/8501 服务；用户可分别启动对比。
网页的置信度和 NMS 滑块仍生效。CPU 模式用 NMS 0.70 获取上述 mAP；若误报多可比较 0.30，并重新检测。
关闭两个新终端即可停止试用；原服务入口依然可用。
PowerShell 如因执行策略阻止 ps1，可直接运行：

```powershell
& 'E:\yolo_workspace\Scripts\python.exe' -m uvicorn accuracy_api:app --host 127.0.0.1 --port 8001 --workers 1
```

此 API 复用 E:\yolo_project\onnx_inference_api_pluggable_v4.py 的接口结构与 DeepSeek 分析逻辑。
DeepSeek 仍按原环境变量启用；模型验证期间关闭了 DeepSeek，没有发送图片或检测结果到外部模型。
可通过 STEEL_PROJECT_ROOT、FUSION_MODEL_A_PATH、FUSION_MODEL_B_PATH 指定其他本地项目或 ONNX 路径。

### 3. 只更换单模型

新权重为 models/texture_ft_best.pt，导出权重为 models/texture_ft_best.onnx。
如需要原架构的单模型推理，可直接加载这份新权重，无需使用融合脚本。
相同官方验证配置（imgsz=640、rect=False、NMS=0.70、conf=0.001、batch=8、GPU）：

- 原 mosaic000：mAP@0.5=70.1787%，mAP@0.5:0.95=36.0526%。
- 新 texture_ft：mAP@0.5=72.5441%，mAP@0.5:0.95=36.9988%。
- 改善分别为 2.37 和 0.95 个百分点。
- 新模型没有增加网络规模；融合/TTA 的精度提升则需要额外推理计算。

## 微调与验收

从已有 mosaic000 best.pt 开始，以 AdamW lr0=0.0002、cos_lr=True、640 输入进行有限微调。
增强假设是减弱缩放、平移和亮度扰动以保留灰度纹理：scale=0.15、translate=0.05、hsv_v=0.15、hsv_h=hsv_s=0，保持 Mosaic 关闭。
这是组合增强策略的验证，不能把收益归因于某一个独立参数。
seed=42，最多 40 epoch，patience=12；实际完成 19 epoch，最佳训练验证 mAP50-95 出现在第 7 epoch。
训练结束后重新用部署一致的 square/rect=False 设置进行官方验证，最终采用该复核值。
验收预先规定：mAP50-95 至少超过两个单模型基线 0.005，任一类别 AP50 相对 mosaic000 不下降超过 0.03；结果通过。
弱项龟裂在新单模型上仍略有退步，融合后也未完全解决；氧化铁皮压入仍是较弱类别。

## 评估边界与实现

- 数据逐文件检查未发现缺失标签、非法框或跨训练/验证完全相同的像素图片；训练集有一对完全重复图片，本轮未修改原数据。
- 这次审计未证明没有近似图片或同批次相关性，也未判断每个框的语义质量。
- 项目仍没有独立测试集。模型和推理方案都使用现有验证集比较，结果有选择偏差；不能作为生产泛化证明。
- 原报告中约 72.6%/72.8% 的分数与 square ONNX 配置不一致；本包不会把历史配置与本轮配置混算提升幅度。
- mAP 的 P/R 是官方曲线选点；固定 conf 的 P/R 是另列的一对一计数，二者不混用。
- 融合先执行官方 multi-label NMS 0.70，可选再执行 0.30，然后做两源同类框置信度加权融合。每个融合簇最多接受每个模型一份框，单模型框分数除以 2；最多 300 框。
- 融合 IoU=0.55，候选分数至少 0.01。TTA 是两个 PyTorch 模型各自的原生 TTA，ONNX 模式不伪装支持 TTA。
- 0.30 配置是两次顺序 NMS 的明确实现，与直接在全部原始候选上做一次 0.30 NMS 不保证完全相同。
- 缓存重算 AP 与原生 Ultralytics val 在四个原模型配置上通过数值一致性检查；最终 CPU/GPU 部署实现均重新推理了全部 360 张。
- API 已通过健康检查、实际图片推理、坐标范围、阈值、损坏图片和非法 IoU 测试。未进行云端部署或全网页浏览器测试。
- API 测试需要本地套接字；沙箱内 Windows socketpair 被阻止，最终测试在获批的沙箱外环境执行通过。

环境：Python 3.10、PyTorch 2.5.1+cu121、Ultralytics 8.4.177；使用现有 E:\yolo_workspace 环境，未升级依赖。
模型、预测样例、训练参数和结果见 models、sample_prediction、evidence。
本包保留实验脚本用于复现；重新训练会在执行目录创建 accuracy_experiments，不会覆盖原项目 runs。

参考：[Ultralytics 验证配置](https://docs.ultralytics.com/modes/val/)、[数据增强配置](https://docs.ultralytics.com/guides/yolo-data-augmentation/)。
