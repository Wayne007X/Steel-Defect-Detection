# NEU-DET 新评估协议与受控重训

协议版本：`neu-rebuild-20261031-v1`。用途：论文／答辩复核。

本项目此前已经使用 NEU-DET 进行探索。新流程用于重新训练及内部评估，不宣称相似性分组是真实钢板/批次分组，也不宣称新划分抹除了历史选择信息。

## 当前交付状态

已完成源数据审计、1,800 张原图/标签的独立副本、19 对候选的 pHash/SSIM 与对照材料、固定版本划分工具、官方 COCO 初始化权重核验，以及阶段命令和自动检查。

**尚未导入人工复核决定，未生成新划分，未冻结协议，未启动任何 NEU-DET 重训，未开封 Test。** 这不是训练结果报告。

源数据为 `D:\yolov8\NEU-DET`，原项目和既有服务不覆盖。独立目录内的 `dataset/images` 与 `dataset/labels` 是逐文件校验的原始副本，没有增强或标签纠正。

## 第一步：人工分组复核

请打开 `review/人工分组复核.md`；两张总览为 `review/contact_01.jpg`、`review/contact_02.jpg`，19 张单对大图在 `review/pairs`。

- P001 是完全相同的像素图片，固定同组；两份原标签都保留，不自行挑选其中一份。
- P002–P019 共 18 对需要你判断。可以直接在聊天中逐对回复“同源或近重复／不同／无法确定”，并补充理由；也可以编辑复核 JSON。
- “无法确定”保守合组，但报告会明确它不是确定的来源证据。
- 同一图片组由连接关系合并后会完整展示；如果“不同”的两张通过其他边连成同组，程序阻断，不忽略冲突。

若编辑文件，复制 `review/decisions_template.json` 为 `review/decisions.json`，填写 reviewer、reviewed_at 和所有 decision/reason。decision 使用 `same`、`different`、`uncertain`。不要保留 null，不要改 pair_id 或候选 SHA256。

进入本目录运行：

```powershell
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py import-review --file .\review\decisions.json
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py propose-split
```

导入的复核记录不可静默覆盖。若确认后需要改组或修改协议，应新建协议版本，保留旧记录。

## 第二步：查看划分审计并确认冻结

划分工具使用隔离安装的 scikit-learn 1.5.2，StratifiedGroupKFold 五折、shuffle=True、seed=42，输入按主类别与图片数字编号排序。第 0 折 Test，第 1 折 Val，第 2–4 折 Train。

图片主类别与实际目标类别分别统计：数据确有同一张图包含不同类别 GT 的情况，不会按文件夹名强制改标签。
另有 3 张原图的标签各包含 1 行完全重复框。磁盘原标签保留；固定版本 Ultralytics 会在内存中去除重复行，因此全数据原始框数为 4189，实际读取唯一框数为 4186。审计和划分报告同时列出两种口径，避免把读取规则差异当成模型漏检。冻结前请一并确认这项处理记录。
目标约 1080/360/360，组完整性优先，不根据模型分数换 seed、拆组或重新抽样。

`split/划分审计.md` 会给出实际分布、组成员、主类别/目标类别计数，以及 dataset 中 train.txt、val.txt、test.txt 的 MD5 和 SHA256。
训练 YAML 只有 Train/Val，没有 Test 键。Test TXT 用于明确冻结成员，不会被训练入口读取。

你确认审计及非单例组后，使用报告中的完整 Proposal SHA256：

```powershell
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py approve-split --proposal-sha '报告中的完整SHA256' --reviewer '实际确认人'
```

缺少复核、文件变化、来源校验失败或 SHA 不对应时均会拒绝。第一份冻结记录固定代码、协议、划分、数据清单、初始化权重及训练环境。

## 第三步：最多 8 小时的串行训练与评估

初始化权重来自固定官方发布地址：

`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt`

官方文件 SHA256 为 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`，元数据为 COCO 80 类，与原本地通用权重逐字节一致。初始化使用本目录核验后的 models/yolov8n_coco.pt，不使用历史 NEU-DET 权重。

固定种子 42/7/123，每组配方各三个任务：

- A：标准增强，Mosaic=1，最后 10 轮关闭；从通用预训练开始。
- B：与 A 相同，只关闭 Mosaic。
- C：从新 B 的同种子最佳检查点进行温和纹理增强微调。
- D：从与 C 完全相同的 B 检查点、使用相同微调预算，但保持 B 原增强，控制额外训练的影响。

基础训练最多 100 epoch、patience=20；C/D 最多 40、patience=12。完整参数在 protocol.json 中显式固定。
训练 AMP 开启，验证强制 FP32、square 640、batch=1。早停与 best.pt 都按 Val AP50，严格提升才更新，同分保留早期检查点；不能默默恢复库默认 AP95 标准。
发生显存问题时不允许自动减 batch 后继续冒充同一实验，需检查并报告。

```powershell
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py train
```

仅在检查中断原因后，使用 `train --resume` 续跑相同任务。它不允许指定外部权重，C/D 初始化 SHA 必须匹配新 B。
计算任务通过 OS 文件锁串行执行，预算日志保存在 state/budget.json；训练部分使用前 7 小时，最后 1 小时留给选型、导出与验证，累计总上限 8 小时。
预算按实际计算任务计时，人工等待不计；进程异常用最后一秒心跳保守记账，不按关机后等待数天计费。达到预算后保存已有检查点并停止，任务未齐时不能选型或开 Test。
预算在训练批次、验证批次及逐图推理边界检查；不能承诺中断一个正在执行的 CUDA 内核的毫秒级时间点。

## 第四步：Val 选型、实际推理复核和导出

```powershell
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py select-val
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py verify-export
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py freeze-final
```

只有 12 个任务完成、检查点和初始化来源链一致时才运行。
候选固定为单模型 A/B/C/D、双模型 A+B/A+C；TTA 开/关；NMS 原生 0.70、顺序 0.30、弱类 0.50/其余 0.30。共 36 个配置，每个三个种子，共 108 个配置/种子条目。
融合等权、IoU=0.55、候选分数下限 0.01；共享工作点阈值只搜索 0.10–0.50、步长 0.05。不额外搜索类别置信度或融合权重。

研究方案按三种子平均 AP50 及预设晋级门槛选择；未达门槛则保留 B。平衡方案选平均 F1；召回方案在 P/F1 损失约束下选 Recall。全部阈值与选择理由写入 Val 报告。
不挑一个最高分种子当唯一成绩；部署示例固定 seed42。

实际推理在 Val 上重跑已选配置，并与原生缓存指标对照。速度报告预热后中位数/p95、设备与 batch，不直接承诺流水线速度。
ONNX 固定 FP32、640、batch=1、opset=17，AP50/AP95 误差均不超过 0.002 才标记对应 ONNX 变体可部署。
**ONNX 不执行 TTA。** 若平衡 GPU 配置带 TTA，导出报告明确验证其无 TTA 分支，不将其称为 TTA 方案的等效部署。

## 第五步：人工确认最终冻结报告后统一 Test

`freezes/最终冻结报告.md` 与 final.json 包含冻结 ID、Val 选择、权重/数据/代码哈希、预登记测试行和三个种子。
你明确确认后才执行：

```powershell
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py approve-test --freeze-sha '最终冻结报告中的完整SHA256' --reviewer '实际确认人'
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py test
```

统一测试 A/B/C/D 原生单模型、A+C 的四个融合消融，以及 Val 选出的研究/平衡/召回配置，重复的配置和阈值去重。
所有行都报告三个种子、均值、样本标准差、逐类 AP 与工作点 TP/FP/FN、P/R/F1。不会以 Test 再调参数或选种子。
中断只能按相同冻结 ID 续跑，不得换方案后仍称首次统一测试。缓存附权重、划分和代码 SHA；不接受其他协议或变更后的缓存。

## 检查与维护

```powershell
& 'E:\yolo_workspace\Scripts\python.exe' .\run_protocol.py status
& 'E:\yolo_workspace\Scripts\python.exe' -m unittest discover -s .\tests -v
```

自动检查使用隔离临时合成数据和模拟训练器验证协议逻辑，不进行 NEU-DET 重训或提前运行 Test。
训练环境保持原版本；划分依赖只装在 tools/split_deps，由独立 worker 进程加载，不升级训练环境。
框架缓存、AMP 检查辅助权重和日志限定在本协议目录；外部实验跟踪与同步关闭，不修改用户的全局设置或复制全局密钥。
如需在相同环境重新准备依赖，运行 setup_split_dependencies.ps1。不要移动冻结后的目录，训练 YAML 和检查点路径已经纳入冻结记录。

这些命令是科学实验流程守卫，不是操作系统级数据保密系统。原数据曾被探索过，该历史在每份最终报告中披露。
PACKAGE_MANIFEST_SHA256.json 仅校验初始交付的静态代码、数据副本、依赖和复核材料，不代替之后的划分冻结或最终模型冻结；使用 check_delivery.py 可检查静态交付完整性。
