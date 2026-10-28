# 钢材表面缺陷检测系统：部署架构与端到端回归记录

> 文档用途：作为本地 ONNX 推理应用的部署说明和回归测试记录。
> 当前仓库：`Wayne007X/Steel-Defect-Detection`
> 本地项目目录：`E:\yolo_project`
> 本地 Python 环境：`E:\yolo_workspace`
> 文档更新日期：2026-10-28

## 1. 当前结论

项目已经从模型实验推进到可交互的本地应用阶段。当前已验证的主要链路为：

- YOLOv8 六类钢材表面缺陷模型训练、评估和候选模型比较；
- PyTorch 权重导出为 ONNX，并进行推理一致性/验证流程检查；
- FastAPI 加载可配置 ONNX 模型，提供健康检查、视觉检测和可选 AI 分析接口；
- Streamlit 通过 HTTP 调用 FastAPI，展示检测框、类别、置信度、图片尺寸和推理耗时；
- DeepSeek 作为可选辅助分析服务，失败或未配置密钥时不应影响已经完成的视觉检测。

**当前本地应用的正式后端入口应使用 `onnx_inference_api_pluggable_v4.py`。** V4 已加入 CPU-only 初始化、关闭 Ultralytics 自动安装、CPU 预热和显式 `device="cpu"`。V3 可作为历史版本备份，不应继续作为本地应用的默认启动入口。

本记录代表开发环境中的功能回归通过，不等于已完成工业现场验收、独立测试集评估或生产级安全审计。

## 2. 系统架构

```text
用户浏览器
   │
   ▼
Streamlit 前端
   │  POST /predict（上传图片 + conf/iou）
   ▼
FastAPI（onnx_inference_api_pluggable_v4.py）
   │  图片解码、参数校验、调用模型
   ▼
Ultralytics ONNX wrapper
   │  ONNX Runtime CPUExecutionProvider
   ▼
结构化检测 JSON
   │  类别、置信度、原图像素 xyxy 坐标、耗时
   ▼
Streamlit 绘制检测框并展示结果

可选独立分支：
Streamlit ── POST /analyze（只发送检测 JSON）──> FastAPI ──> DeepSeek API
```

### 2.1 模块职责

| 模块 | 职责 | 边界 |
|---|---|---|
| 模型权重（`.pt` / `.onnx`） | 提供缺陷检测能力 | 权重文件不应随普通源码提交到 Git；候选 `.pt` 已通过 GitHub Releases 分发 |
| `onnx_inference_api_pluggable_v4.py` | 加载 ONNX、提供 `/health`、`/predict`、`/analyze` 等接口 | 模型路径由后端环境变量控制；当前正式运行模式为 CPU |
| Streamlit 前端 | 上传图片、调整阈值、显示检测结果和 AI 分析 | 不直接加载 PyTorch、Ultralytics 或 ONNX 模型，只通过 HTTP 调用 FastAPI |
| `verify_fastapi_live.py` | 对正在运行的 API 做实时冒烟测试 | 测试需要指定一张存在的图片；`--test-analyze` 可能只验证未配置密钥时的降级契约 |
| `scripts/`、`experiments/`、`reports/` | 保存训练、评估、诊断脚本及实验材料 | 应保留有复现价值的代码、指标与报告，不提交本地运行缓存和重复打包副本 |

### 2.2 主要接口

- `GET /health`：报告服务状态、模型标签、设备、输入尺寸、六类类别名和 ONNX Runtime 可用 provider。
- `POST /predict`：接收图片和推理参数，返回检测数量、类别 ID/名称、置信度、原图像素 `xyxy` 坐标、图片尺寸和推理耗时。
- `POST /analyze`：可选地把 `/predict` 的结构化 JSON 发送给 DeepSeek，返回辅助解读。该接口不会把原始图片传给 DeepSeek，因此 AI 分析不能声称直接看过原图。
- `/docs`：FastAPI Swagger 交互文档。

### 2.3 CPU 推理说明

本地环境日志显示 ONNX Runtime 可用 provider 包含 `AzureExecutionProvider` 与 `CPUExecutionProvider`，没有 `CUDAExecutionProvider`。V4 的运行策略是显式 CPU 推理，不需要为当前服务反复尝试安装 `onnxruntime-gpu`。

V4 的相关实现包括：

1. 在导入 Ultralytics 前设置 `YOLO_AUTOINSTALL=false` 和 `CUDA_VISIBLE_DEVICES=-1`；
2. FastAPI 启动期间用内存中的空白图片执行 CPU 预热；
3. 预热和实际 `/predict` 都显式传入 `device="cpu"`；
4. 启动阶段完成 ONNX backend 初始化，避免首次用户请求承担大部分初始化等待。

`GET /json/version` 返回 404 不是本项目的必要 API；只要 `/health`、`/predict` 和 `/docs` 正常，不应把这个独立路径的 404 当作服务失败。

## 3. 模型实验背景

项目以 NEU-DET 六类钢材表面缺陷数据集为基础，当前记录的数据划分为 1800 张图像，其中 1440 张训练图像、360 张验证图像。

| 类别 ID | 英文类别 | 中文名称 |
|---:|---|---|
| 0 | `crazing` | 龟裂 |
| 1 | `inclusion` | 夹杂 |
| 2 | `patches` | 斑块 |
| 3 | `pitted_surface` | 麻点 |
| 4 | `rolled-in_scale` | 氧化铁皮压入 |
| 5 | `scratches` | 划痕 |

当前保留两个用途不同的模型候选（指标来自现有验证集，不是独立测试集）：

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 当前定位 |
|---|---:|---:|---:|---:|---|
| `neu_defect_clean` | 0.673 | 0.650 | 0.7281 | 0.3844 | 整体 mAP 基线 |
| `neu_defect_mosaic000` | 0.664 | 0.689 | 0.7260 | 0.3784 | 召回率优先候选 |

模型不是简单的绝对优劣关系。是否采用召回率优先方案，应结合漏检和误报的业务成本；不要仅凭一张演示图片判断模型整体质量。

## 4. 环境与依赖

已验证的本地环境信息：

- Windows PowerShell
- Python `3.10.11`
- PyTorch `2.5.1+cu121`（当前 FastAPI ONNX 服务仍使用 CPU）
- Ultralytics `8.4.177`
- ONNX Runtime `1.23.2`
- 现有虚拟环境：`E:\yolo_workspace`

前端依赖文件：`streamlit_frontend_requirements.txt`。
仓库整理后，后端 API 附加依赖文件统一命名为 `requirements_api_v4.txt`。该文件列出 API 层依赖和 V4 预热所需的 NumPy，但不负责安装 PyTorch、Ultralytics 和 ONNX Runtime 等核心推理依赖。

在已配置好的本地环境中安装/补齐 API 与前端依赖：

```powershell
cd E:\yolo_project
E:\yolo_workspace\Scripts\Activate.ps1
python -m pip install -r .\requirements_api_v4.txt
python -m pip install -r .\streamlit_frontend_requirements.txt
```

不要为了当前 CPU-only 后端安装 `onnxruntime-gpu`。如果在全新机器上部署，需要另行安装与 Python、Ultralytics、ONNX 模型兼容的 `onnxruntime` CPU 版本及核心模型依赖，并在部署环境实际验证。

## 5. 启动方式

建议本地 FastAPI 使用端口 `8001`，避免与已占用的 `8000` 端口冲突。FastAPI 和 Streamlit 必须在两个独立的 PowerShell 终端中运行，并保持各自终端开启。

### 5.1 终端 A：启动 FastAPI（默认 mosaic000）

```powershell
cd E:\yolo_project
E:\yolo_workspace\Scripts\Activate.ps1

$env:MODEL_PATH = "E:\yolo_project\runs\neu_defect_mosaic000\weights\best.onnx"
$env:MODEL_NAME = "mosaic000"

# 只有需要 DeepSeek 辅助分析时才设置；不要把真实密钥写入源码或提交到 Git。
# $env:DEEPSEEK_API_KEY = "你的真实 API Key"

python -m uvicorn onnx_inference_api_pluggable_v4:app --host 127.0.0.1 --port 8001 --workers 1
```

看到 `Application startup complete` 和 `Uvicorn running on http://127.0.0.1:8001` 后，打开**另一个终端**验证：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/health" -TimeoutSec 10 | ConvertTo-Json -Depth 5
```

预期关键字段：`status=ok`、`model=mosaic000`、`device=cpu`、`imgsz=640`，且 `class_names` 包含六类。

### 5.2 终端 B：启动 Streamlit

当前已验证的前端代码已统一命名为 `streamlit_frontend.py`。仓库整理脚本会先比较旧文件与测试版本，并将旧文件备份到仓库目录之外，再将测试版本设为正式入口：

```powershell
cd E:\yolo_project
E:\yolo_workspace\Scripts\Activate.ps1
$env:FASTAPI_BASE_URL = "http://127.0.0.1:8001"
python -m streamlit run .\streamlit_frontend.py
```

浏览器打开终端打印的 Local URL，通常为 `http://localhost:8501`。如果更换 FastAPI 端口，也必须同步修改 `FASTAPI_BASE_URL`。

### 5.3 切换到 clean 模型

先停止当前 FastAPI 进程（在运行服务的终端按 `Ctrl+C`），再在新的服务进程中设置：

```powershell
$env:MODEL_PATH = "E:\yolo_project\runs\neu_defect_clean\weights\best.onnx"
$env:MODEL_NAME = "clean"
python -m uvicorn onnx_inference_api_pluggable_v4:app --host 127.0.0.1 --port 8001 --workers 1
```

切换模型后，必须再次检查 `/health`，并在冒烟测试中显式指定 `--expected-model clean`。不要在服务运行过程中只修改 PowerShell 环境变量并期待已加载的模型自动切换；需要重启服务进程。

## 6. 回归测试记录

### 6.1 FastAPI 冒烟测试

测试图片：`D:\yolov8\NEU-DET\validation\images\crazing\crazing_253.jpg`（200 × 200）。

| 测试项 | 已记录结果 | 说明 |
|---|---|---|
| mosaic000 基本 API 冒烟测试 | `14/14 passed` | `/health`、`/predict` 契约、尺寸、框坐标、类别、置信度和异常输入校验通过 |
| mosaic000 含 `/analyze` 契约测试 | `16/16 passed` | 额外验证 `/analyze` 结构化降级响应及之后 `/predict` 仍可用 |
| clean 基本 API 冒烟测试 | `14/14 passed` | 命令显式传入 `--expected-model clean` 后全部通过 |
| V4 CPU 后端启动 | 已验证 | 日志显示 CPU warm-up 完成，推理设备明确为 CPU |
| Streamlit 端到端交互 | 已从截图确认 | 页面能连接 FastAPI、显示检测框及概览，并显示 AI 辅助分析结果 |

mosaic000 在该测试图片上的一次记录：

- 检测数：2，类别均为 `crazing`；
- 置信度约为 `0.546265` 和 `0.266782`；
- 一次已预热推理耗时约 `25.829–40.429 ms`。这是单张 200 × 200 图片的观测值，不是固定性能承诺。

clean 在同一张图片上的一次记录：

- 检测数：1，类别为 `crazing`；
- 置信度约为 `0.46962`；
- 一次已预热推理耗时约 `31.534 ms`。

### 6.2 测试命令

如果 `verify_fastapi_live.py` 仍位于项目根目录：

```powershell
python .\verify_fastapi_live.py `
  "D:\yolov8\NEU-DET\validation\images\crazing\crazing_253.jpg" `
  --base-url "http://127.0.0.1:8001" `
  --expected-model mosaic000 `
  --test-analyze
```

若将它整理到 `scripts\testing\verify_fastapi_live.py`，命令相应改为：

```powershell
python .\scripts\testing\verify_fastapi_live.py `
  "D:\yolov8\NEU-DET\validation\images\crazing\crazing_253.jpg" `
  --base-url "http://127.0.0.1:8001" `
  --expected-model mosaic000 `
  --test-analyze
```

`--test-analyze` 可能在没有 `DEEPSEEK_API_KEY` 时验证的是 `status=unavailable` 的降级契约。因此，`16/16` 不应单独表述为“这次测试实际成功调用了 DeepSeek”。另外，用户提供的后续 Streamlit 截图确实显示 AI 分析完成；两者对应的进程配置/运行时段不同，交付时应分别说明。

### 6.3 已通过的输入校验

- 空文件：HTTP 400；
- 无法解码的非图片文件：HTTP 400；
- 超过 10 MiB 的图片：HTTP 413；
- 无效置信度参数：HTTP 422；
- `/predict` 返回后，检测数量与检测条目数一致；
- 检测框坐标合法，且采用原图像素坐标；
- 类别 ID 与 `/health` 的类别名一致；
- 置信度在 `[0, 1]` 范围内。

## 7. 尚未充分验证的事项

- [ ] 对六个类别分别选取多张图片做前端与 API 回归，而非只测试一张 `crazing` 图片。
- [ ] 专门模拟 DeepSeek 超时、断网和服务端错误，验证 UI 降级时视觉检测结果仍保留。
- [ ] 清理仓库后重新执行一次完整 API 冒烟测试和 Streamlit 手工检查。
- [ ] 仓库整理后重新启动 `streamlit_frontend.py`，确认统一文件名没有影响前端功能。
- [ ] 确认 clean 模型切换后的 Streamlit 页面联动。
- [ ] 独立测试集、真实生产批次、不同光照/相机条件、吞吐量与工业现场验收仍不在现有测试范围内。

## 8. 仓库整理原则

建议最终仓库保留源码、可复现脚本、实验指标、正式报告和部署文档；本地权重、数据集、缓存和重复打包产物不放进普通源码提交。

建议的组织方向：

```text
.
├── README.md
├── DEPLOYMENT_AND_REGRESSION.md
├── onnx_inference_api_pluggable_v4.py
├── streamlit_frontend.py            # 统一后的正式前端入口
├── requirements_api_v4.txt
├── streamlit_frontend_requirements.txt
├── scripts/
│   ├── analysis/
│   ├── diagnostics/                 # 诊断/排查脚本
│   ├── evaluation/                  # 模型验证和 ONNX 一致性检查
│   ├── postprocessing/
│   ├── testing/                     # verify_fastapi_live.py
│   └── training/
├── experiments/neu_det_tuning/
├── deployment_model_comparison/
└── reports/
```

整理时遵循以下规则：

1. 不删除已验证通过的代码；先比较文件哈希和差异，再决定是否去重。
2. `EXPERIMENT_INDEX.md` 已存在于 GitHub；若本地 `experiment_index_updated.md` 内容完全相同，不要把重复副本再次提交。
3. V3/V4 的 zip、bundle 和 package 目录属于重复分发副本，除非有明确发布需求，不要和正式源码一起提交；可保留在本地备份，或改用 GitHub Releases 分发。
4. `runs/`、`.pt`、`.onnx`、本地数据集、缓存、虚拟环境和密钥不应进入源码提交。
5. `experiments/`、`reports/`、CSV 指标与必要的复现脚本属于项目成果，不要用宽泛的忽略规则将其排除。
6. 提交前检查 `git status --short` 和 `git diff --cached --stat`，并逐项审查 `git diff --cached`；不要直接 `git add .`。

## 9. 最终小结

当前已完成模型候选比较、ONNX 推理路径、FastAPI 接口、Streamlit 可视化和可选 DeepSeek 辅助分析的主要开发链路。V4 CPU-only 后端和 API 冒烟测试已在本地通过。下一阶段的重点不是重写推理代码，而是统一正式文件入口、补齐部署文档、去除重复打包副本、完善 `.gitignore`，然后分批提交并推送到 GitHub。
