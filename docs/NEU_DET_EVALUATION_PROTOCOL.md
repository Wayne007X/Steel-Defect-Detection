# NEU-DET 新评估协议与受控重训方案

本文件说明预先约定的规则及实际执行状态；不覆盖原冻结配置。最终状态：12项训练完成，Val选型冻结，用户确认后统一Test完成；C/D执行偏差已在开封前披露。

## 目标、预算和范围

用于论文／答辩复核，精度优先兼顾速度；仅使用现有NEU-DET。最多8小时串行训练及模型评估，人工等待不计；训练上限7小时，预留1小时。达到预算未完成全部任务时不开放Test。累计至统一Test的登记计算约5.105小时。

不更换架构、不改原标签、不增加Copy-Paste、MixUp或亮度归一化。旧实验与服务保留。披露之前已使用NEU-DET探索，不能称为新来源外部盲测。

## 数据审计与人工分组

1. 汇总原Train/Val共1800图及标签，核验尺寸、类别、框有效性、XML/YOLO一致性，保存SHA256。
2. 相同像素固定同组；重复图及两份原标签均保留。
3. 现有63位pHash全量两两比较，Hamming≤12产生19对候选，附SSIM和原图对照；相似性不证明真实来源。
4. 用户逐对判断同源/近重复、不同、无法确定；无法确定保守同组。检查完整连通组及传递矛盾。
5. 实际P001像素重复；P002/P003/P009/P015/P016人工近重复；P005无法确定保守同组；其余不同。形成1793组，其中7个双图组。

主类别每类300图，123张图包含与文件名主类不同的目标类；划分按主类分层，实际目标分布另报。原4189框、读取有效4186框，原始标签字节不变。

## 划分与第一次冻结

固定scikit-learn 1.5.2，StratifiedGroupKFold五折、shuffle=True、seed42；按类别/图号固定排序，第0折Test，第1折Val，其余Train。组完整性优先，不按模型分数重抽，也不拆组凑数。

实际Train/Val/Test=1081/360/359，全部覆盖六类；有效目标2520/808/858。保存清单MD5/SHA256及组成员、人工决定、主类/目标分布。人工确认审计后冻结；改动须新版本。训练YAML仅有Train/Val，Test命令单独守卫。

[划分审计](../evaluation_rebuild_20261031/split/划分审计.md) · [清单哈希及分布](../evaluation_rebuild_20261031/split/proposal.json) · [第一次冻结](../evaluation_rebuild_20261031/freezes/split.json)

## 初始化、训练与检查点

YOLOv8n从固定官方COCO通用预训练开始，核验80类元数据、来源和SHA256。A/B不使用旧clean、mosaic000、texture等NEU权重。种子固定42/7/123。

- 共同：640、batch16、GPU0、workers0、AMP、deterministic；box/cls/dfl=7.5/0.5/1.5，momentum0.9、weight_decay0.0005、warmup_bias_lr=0。
- 基础：AdamW，lr0=0.001、lrf=0.01、线性衰减、warmup3；最多100 epoch，patience20。
- A：Mosaic1，最后10轮关闭；HSV0.015/0.7/0.4、scale0.5、translate0.1、fliplr0.5。
- B：与A相同，只关闭Mosaic。
- C：同种子新B最佳权重起点，HSV0/0/0.15、scale0.15、translate0.05。
- D：与C相同B起点，保持B的HSV/scale/translate，采用与C相同微调优化器与预算。
- C/D微调：AdamW，lr0=0.0002、lrf=0.1、余弦衰减、warmup2；最多40 epoch，patience12。按A三种子、B三种子、C三种子、D三种子串行执行。
- 验证：square640、rect=False、FP32、batch1、候选conf0.001、NMS0.70、max_det300。best和早停均使用Val AP50；严格提高才更新，相同保留较早轮次。

**原计划与实际偏差**：原文字称C/D“其余几何增强关闭”，实际继承共同设置fliplr=0.5。六个任务均如此，几何旋转/错切/透视/上下翻转及图像混合关闭。用户在Test前决定不重训，报告按实际配置说明；C/D相同翻转不代表翻转无影响。原冻结不追溯改写。

[实际机器配置](../evaluation_rebuild_20261031/protocol.json) · [偏差说明及完整Val报告](../evaluation_rebuild_20261031/reports/supplement_v1/Val结果与执行偏差补充报告.md)

## 有限Val候选池与选择规则

主指标是三个种子的平均Val AP50，同时报告AP50:95、逐类AP及工作点计数。

- 家族：A/B/C/D、A+B、A+C；双模型按相同种子配对。
- TTA开/关；NMS原生0.70、顺序0.70→0.30、弱类第二级0.50/其余0.30。
- 等权融合、IoU0.55、候选floor0.01、每簇每模型最多一个框；不搜融合权重。
- 工作点置信度仅0.10–0.50、步长0.05；三个种子共享阈值，不搜索类别阈值。
- 研究方案：平均AP50最高；差≤0.0001时按AP50:95更高、前向次数更少、配置编号更小依次择优。相对B晋级需AP50至少+0.005、AP50:95损失≤0.005、任一类AP50损失≤0.03；无通过则保留B。
- 平衡方案：平均F1最高，并列优先较低计算量、较高阈值。
- 召回方案：P不低于平衡模式3个百分点、F1不低于1个百分点，选最高平均Recall；并列优先F1及较少计算量。
- 不挑最佳种子当唯一成绩；部署示例固定seed42。

实际36配置×3种子=108条，每条9档工作点。[完整网格](../evaluation_rebuild_20261031/val/grid.json)和[选择结果](../evaluation_rebuild_20261031/val/selection.json)均保留。

## 最终冻结与Test

冻结模型SHA256、数据/代码/环境、Val结果与选择理由、导出验证、全部预登记测试行。用户确认补充披露材料后才开封。

Test预登记：A/B/C/D原生单模型；A+C无TTA融合、TTA融合、全局0.30、类别NMS；加Val选出的研究/平衡/召回方案，重复配置与阈值组合去重。最终11行×3种子=33条，包含8种预测配置。前8项对照工作点0.25；选定研究0.50、平衡/召回0.35。

AP用低候选阈值完整预测计算；P/R/F1在固定阈值单独计算；报告种子结果、均值、样本标准差、逐类AP与TP/FP/FN。硬件中断仅按同一冻结续跑；Test不选阈值或改方案。根据Test诊断的后续改动只能记为新探索。

- [原最终冻结报告](../evaluation_rebuild_20261031/freezes/最终冻结报告.md)
- [原最终冻结清单](../evaluation_rebuild_20261031/freezes/final.json)
- [补充封存](../evaluation_rebuild_20261031/reports/supplement_v1/supplement_freeze.json)
- [Test完整结果](../evaluation_rebuild_20261031/test/Test统一评估报告.md)
- [Test归档清单](../evaluation_rebuild_20261031/test/archive_manifest.json)

## 一致性、部署与限制

阶段入口为审计→人工复核→划分冻结→训练→Val选型→最终冻结→授权Test，守卫检查前置状态与哈希；不能回退到Val冒充Test。

原生缓存与实际逐图推理核对；ONNX固定FP32、640、batch1、opset17，两项mAP绝对差≤0.002才标为对应变体可部署。ONNX不执行TTA。速度仅Val实测，记录中位数/p95；CPU型号和历史温度未记录，不能补造。

原始库和本地大型产物不随普通Git提交；仓库中的哈希指向完整本地实验档案。复核与移植说明见[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

报告统一表述：**披露历史探索及执行偏差后，在新固定协议下重新训练并评估的NEU-DET内部泛化结果；外部生产泛化能力仍待验证。**
