# 代码、配置与核验入口

## 只读复核：默认入口

在仓库根目录用Python 3.10+运行：

```powershell
python .\scripts\testing\verify_repository_evidence.py
```

默认repository范围只用标准库，检查本地整理时封存的可提交证据文件SHA256、Val/Test冻结关联、33条结果与均值/样本标准差和逐类计数。没有训练、推理、联网或Test开封操作；不是重新测量模型精度。

原开发机具有全部本地资产和原依赖时，可选：

```powershell
& 'E:\yolo_workspace\Scripts\python.exe' .\scripts\testing\verify_repository_evidence.py --scope full
```

full额外调用原冻结/来源/环境/完整归档检查，缺少原图、权重、缓存或固定依赖时拒绝，不静默降级为repository。

## 当前研究主入口

- [run_protocol.py](../evaluation_rebuild_20261031/run_protocol.py)：审计、分组导入、划分提案/批准、训练、Val、最终冻结、Test的阶段命令。
- [protocol.json](../evaluation_rebuild_20261031/protocol.json)：实际固定配置；不是事后修订成理想方案。
- [core.py](../evaluation_rebuild_20261031/neu_protocol/core.py)：哈希、冻结、模型来源、预算和阶段守卫。
- [data.py](../evaluation_rebuild_20261031/neu_protocol/data.py)：审计、分组决定和固定划分；[split_worker.py](../evaluation_rebuild_20261031/neu_protocol/split_worker.py)隔离固定划分依赖。
- [training.py](../evaluation_rebuild_20261031/neu_protocol/training.py)：A/B/C/D、AP50最佳权重与早停。
- [prediction.py](../evaluation_rebuild_20261031/neu_protocol/prediction.py)、[fusion_reference.py](../evaluation_rebuild_20261031/neu_protocol/fusion_reference.py)：统一预测、融合与指标。
- [evaluation.py](../evaluation_rebuild_20261031/neu_protocol/evaluation.py)：有限Val池、选型、ONNX一致性及预登记Test。
- [协议实现测试](../evaluation_rebuild_20261031/tests/test_protocol.py)：原实现检查，使用合成/临时数据，不触发实际NEU重训。

这些源文件仍在原位置，保留原SHA；本次不重命名或重构被冻结的实现。新环境如需改路径、配方或依赖，要建立新协议版本。

## 当前本地应用入口

- [新版API](../evaluation_rebuild_20261031/post_test_delivery/web/rebuilt_api.py)、[新版前端](../evaluation_rebuild_20261031/post_test_delivery/web/streamlit_rebuilt.py)。
- [run_api.ps1](../evaluation_rebuild_20261031/post_test_delivery/web/run_api.ps1)、[run_frontend.ps1](../evaluation_rebuild_20261031/post_test_delivery/web/run_frontend.ps1)：原开发机启动，端口8002/8503。
- [应用说明](../evaluation_rebuild_20261031/post_test_delivery/web/README_网页接入.md)：模型准备、GPU/CPU、AI开关、回退和停止。

此应用依赖原完整冻结资产，普通克隆不包括新版权重。现有运行实例与启动脚本分开管理，不读取或发布密钥。

## 保留的旧应用与历史脚本

- 根[onnx_inference_api_pluggable_v4.py](../onnx_inference_api_pluggable_v4.py)、[streamlit_frontend.py](../streamlit_frontend.py)：旧CPU/云端路线，继续保留。
- 根[app.py](../app.py)：更早的Streamlit直接加载neu_defect_train原型，不是当前推荐入口。
- [scripts](../scripts/README.md)：早期训练、评估、分析、后处理和诊断；同名修正版/controlled版本不代表多个独立公平实验。
- [accuracy_upgrade](../accuracy_upgrade_20261030/README_精度优化结果与运行方法.md)、[accuracy_round2](../accuracy_round2_20261030/README_第二轮结果与运行方法.md)：前两轮旧Val探索的交付，不使用它们初始化本轮A/B。

## 环境和配置位置

- [依赖用途与限制](../requirements/README.md)、[直接依赖快照](evidence/environment_versions.json)。原根requirements.txt保持前端用途。
- [新协议完整规则](NEU_DET_EVALUATION_PROTOCOL.md)、[跨机器复核说明](REPRODUCIBILITY.md)。
- [本地整理清单](LOCAL_REVIEW.md)：哪些文件拟提交，哪些大型产物保持本地；尚未git add/commit/push。

核验清单记录当前文件哈希，未来正常编辑文档后需要在审查记录中另建新版清单，不应把合法文档更新伪装成原清单一直有效。
