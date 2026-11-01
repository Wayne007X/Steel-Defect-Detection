# 实验与证据索引

建议阅读顺序：项目首页 → 新评估协议 → 最终总结 → 完整Test及执行偏差 → 代码与复核。

## 当前主线

- [完整项目历程](PROJECT_HISTORY.md)：从基线、失败实验到固定协议和工程落地。
- [代码与配置入口](CODE_AND_CONFIG.md)、[待提交审阅](LOCAL_REVIEW.md)：当前主线与历史入口、只读核验及文件范围。

- [项目首页](../README.md)：当前结果、案例和启动入口。
- [NEU-DET新评估协议与受控重训方案](NEU_DET_EVALUATION_PROTOCOL.md)：原约定及实际偏差。
- [最终实验与系统交付总结](../evaluation_rebuild_20261031/post_test_delivery/最终实验与系统交付总结.md)：历史探索、Val、Test、全部消融及部署限制。
- [统一Test完整报告](../evaluation_rebuild_20261031/test/Test统一评估报告.md)、[机器可读结果](../evaluation_rebuild_20261031/test/final_results.json)：11行×3种子，未只保留最高分。
- [Val及执行偏差补充](../evaluation_rebuild_20261031/reports/supplement_v1/Val结果与执行偏差补充报告.md)：C/D水平翻转、三种子与速度口径。
- [弱类别诊断](../evaluation_rebuild_20261031/post_test_delivery/error_review/弱类别错误复核说明.md)：Test后的124张对照，固定seed42平衡模式，不用于本次选型。

## 历史探索（旧Val成绩）

- [第一轮调优报告](../reports/accuracy_tuning_20261030/第一轮调优报告.md)。
- [第二轮调优报告](../reports/accuracy_tuning_20261030/第二轮调优报告.md)。
- [两轮调优综合总结](../reports/accuracy_tuning_20261030/两轮调优综合总结.md)。
- [更早的训练调优记录](../reports/NEU-DET_model_tuning_report.md)。
- [原README快照](history/README_20261030.md)：按当时状态保留，不能当作当前测试状态。

## 划分与冻结

- [数据审计](../evaluation_rebuild_20261031/audit/summary.json)、[人工复核决定](../evaluation_rebuild_20261031/review/approved_decisions.json)。
- [划分审计](../evaluation_rebuild_20261031/split/划分审计.md)、[分布及清单哈希](../evaluation_rebuild_20261031/split/proposal.json)。
- [第一次冻结](../evaluation_rebuild_20261031/freezes/split.json)、[最终冻结](../evaluation_rebuild_20261031/freezes/final.json)、[补充封存](../evaluation_rebuild_20261031/reports/supplement_v1/supplement_freeze.json)。
- [Test授权](../evaluation_rebuild_20261031/freezes/test_authorization.json)、[归档哈希](../evaluation_rebuild_20261031/test/archive_manifest.json)、[交付核验](../evaluation_rebuild_20261031/test/delivery_verification.json)。

## 实现与系统使用

- [阶段入口](../evaluation_rebuild_20261031/run_protocol.py)、[协议实现](../evaluation_rebuild_20261031/neu_protocol/)、[固定配置](../evaluation_rebuild_20261031/protocol.json)。
- [原协议操作说明](../evaluation_rebuild_20261031/README_新评估协议与操作说明.md)：原始交付时的阶段快照，文首“尚未复核”等状态已过时，当前以本索引及最终结果为准；不修改被冻结或交付封存的历史材料。
- [新版网页启动与回退](../evaluation_rebuild_20261031/post_test_delivery/web/README_网页接入.md)、[HTTP/前端验证](../evaluation_rebuild_20261031/post_test_delivery/web_verification.json)。
- [原CPU系统本地指南](../钢材缺陷检测系统_本地启动与使用指南.md)、[早期脚本](../scripts/README.md)。
- [跨机器复核与资产说明](REPRODUCIBILITY.md)。

新权重目前仅保存在本地；旧v1.0.0-models Release不包含本轮A/B/C/D。大型产物由Git忽略，哈希清单是本地完整档案的索引，不是下载地址。
