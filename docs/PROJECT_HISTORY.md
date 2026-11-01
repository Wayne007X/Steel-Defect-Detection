# 项目历程：从缺陷检测原型到固定协议评估

本文件把已有成果按问题与证据串联。历史Val、本轮Val、本轮Test与应用单图展示分开记录；完整原始材料通过链接保留。

## 阶段一：数据整理与YOLOv8基线

早期项目完成NEU-DET图像/标注整理、六类检测训练和图片上传原型。已有实验索引记录过早期标签位置错位及后续纠正，因此更早的neu_defect_train保留为过程证据，不与纠正后的clean/mosaic000作公平比较。

纠正后的clean作为整体mAP基线，mosaic000作为偏召回候选；比较Mosaic、更大YOLOv8s、800分辨率微调、cls/box权重及采样策略。没有达到验收要求的实验仍归档，不能把“有脚本”当作“已验证有效”。

旧官方标准验证中的clean AP50=72.81%、mosaic000=72.60%；后续统一方形输入复核使用另一套评估配置。两种口径不能混合取最高值。

证据：[早期实验索引](../experiments/neu_det_tuning/EXPERIMENT_INDEX.md)、[训练脚本](../scripts/training/)、[早期调优报告](../reports/NEU-DET_model_tuning_report.md)。

## 阶段二：推理后处理与应用落地

完成置信度/NMS扫描、划痕去重、错误诊断和ONNX导出；排查PyTorch与ONNX的推理及验证差异。构建FastAPI的/predict与独立/analyze，Streamlit只调用接口，不在原前端重复加载模型。

DeepSeek仅解释结构化检测输出，不读原图、不改变视觉检测。旧版完成本地AI流程演示及云端纯检测记录；云端关闭AI。历史GIF、检出/漏检案例和部署日志继续保留，不能包装成新版融合模型的性能证据。

证据：[云端部署记录](../cloud_deployment/钢材缺陷检测系统_云端部署过程记录.md)、[ONNX记录](../onnx_fast_api_streamlit_docs/MOSAIC000_ONNX_导出与验证排查记录_补充部署决策.md)、[旧README快照](history/README_20261030.md)。

## 阶段三：两轮精度优先探索

第一轮从统一口径下的旧mosaic000 AP50约70.18%出发，纹理微调seed42达到72.54%；clean+纹理模型+TTA平衡方案在旧Val达到76.71%。CPU无TTA融合也做了实际验证。

第二轮补做seed7/123稳定性复核、弱类错误对照和类别NMS，旧Val召回方案AP50为78.37%，但误报增加。亮度归一化未通过验收，未替换默认方案。

这些结果均来自被反复用于选择的旧Val，不能视为新Test或生产场景成绩。旧Release只含clean/mosaic000，不含后续新协议的A/B/C/D。

证据：[第一轮](../reports/accuracy_tuning_20261030/第一轮调优报告.md)、[第二轮](../reports/accuracy_tuning_20261030/第二轮调优报告.md)、[综合总结](../reports/accuracy_tuning_20261030/两轮调优综合总结.md)。

## 阶段四：重建评估协议与受控重训

审计1800图，以pHash候选、SSIM辅助和人工复核得到相似性组，固定Train/Val/Test为1081/360/359。记录文件和代码哈希，人工确认后冻结；相似性组不等于真实钢板或批次组。

从官方COCO预训练重新训练A/B，再从本轮B同种子检查点得到C/D，各组三种子。最佳权重及早停按Val AP50，保留D以控制额外训练效应。共12任务，按有限候选池进行Val选型。

实际C/D均继承fliplr=0.5，与原文字方案有偏差；在Test前披露并确认不重训，原冻结不改写。C/D对照仍可描述，但不声称完成翻转消融或完全按原文字执行。

证据：[协议](NEU_DET_EVALUATION_PROTOCOL.md)、[划分审计](../evaluation_rebuild_20261031/split/划分审计.md)、[实际配置](../evaluation_rebuild_20261031/protocol.json)、[完整Val/偏差报告](../evaluation_rebuild_20261031/reports/supplement_v1/Val结果与执行偏差补充报告.md)。

## 阶段五：固定方案统一Test

最终冻结及补充材料经确认后，执行11项×3种子，共33条。模型、阈值和种子不按Test修改；研究主方案仍由Val选定，不选择Test最高项来改名为主方案。

研究主方案Test AP50=77.19%±0.50个百分点，AP50:95=44.73%±0.10个百分点。±是训练种子的样本标准差，不是置信区间。平衡/召回的工作点P/R/F1另报，不与AP混算。

披露历史探索后，这些结果用于NEU-DET新固定协议的内部泛化评估；外部产线、相机、光照和正常表面表现仍待验证。与旧Val不同条件，不能用旧78.37%和新77.19%的差来判断方案退步或提升。

证据：[全部Test](../evaluation_rebuild_20261031/test/Test统一评估报告.md)、[原始数值](../evaluation_rebuild_20261031/test/final_results.json)、[Test归档](../evaluation_rebuild_20261031/test/archive_manifest.json)。

## 阶段六：事后诊断与新版本地网页

固定seed42平衡模式的归档Test预测用于事后诊断，124张对照覆盖两弱类的全部漏检/误检。错误描述以分数和重叠为依据，未改标签或本轮参数；后续据此改进只能记为新的探索。

新增独立8002/8503入口，支持GPU平衡/召回、CPU无TTA融合、CPU单模型和旧ONNX回退。用Val样本做真实HTTP及Streamlit交互检查，包括回退后切回GPU；旧服务和云端未替换。三种子均值与网页seed42的单独成绩分别报告。

证据：[最终总结](../evaluation_rebuild_20261031/post_test_delivery/最终实验与系统交付总结.md)、[错误图册](../evaluation_rebuild_20261031/post_test_delivery/error_review/弱类别错误复核说明.md)、[网页验证](../evaluation_rebuild_20261031/post_test_delivery/web_verification.json)。

## 项目目前可以说明什么

项目完成了训练、受控对照、失败实验归档、协议重建、冻结评估、事后诊断与应用部署的完整过程。没有引入新网络架构，不把成熟方法组合自动称为算法创新；更明确的成果是可追溯实验、精度/误报/速度取舍和工程交付。

后续优先准备新来源数据，明确标签规范和业务验收成本。当前仓库不宣称已完成工业现场验收、并发长期运行或外部泛化证明。
