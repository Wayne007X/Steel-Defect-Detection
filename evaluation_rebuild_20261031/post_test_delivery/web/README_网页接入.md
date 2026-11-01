# 新协议 seed42 网页接入

此入口独立于旧服务，使用 8002 / 8503 端口。原 8000/8501 服务及原文件不更改。

## 启动

在 Cursor 中打开两个 PowerShell 终端。第一个运行：

```powershell
& "E:\yolo_project\evaluation_rebuild_20261031\post_test_delivery\web\run_api.ps1"
```

等待后端启动完成后，第二个终端运行：

```powershell
& "E:\yolo_project\evaluation_rebuild_20261031\post_test_delivery\web\run_frontend.ps1"
```

浏览器打开 http://127.0.0.1:8503，确认后端地址 http://127.0.0.1:8002，上传图片并选择方案。

## 检测方案

- GPU 平衡：A+C seed42，TTA，固定第二级 NMS=0.30，默认置信度0.35。
- GPU 召回：相同权重，TTA，弱类第二级0.50、其余0.30，默认0.35。
- CPU 融合：A+C seed42 ONNX，无TTA，默认0.35。不是GPU TTA方案的等效版本。
- CPU C单模型：C seed42 ONNX，无TTA，默认0.25。
- 旧模型回退：原 models/mosaic000.onnx，通过独立入口加载，默认0.25；也可继续使用原网页及服务。

GPU模式在无CUDA时明确报错，不悄悄换成CPU。首次切换模式可能包含模型加载时间。图像检测在本地完成。

首级NMS固定0.70，第二级策略由模式决定。修改置信度属于自定义工作点，不直接对应报告P/R/F1。切换模式会使旧结果失效，需重新检测。

网页固定seed42，不能将三个种子均值标作网页实例精度。最终总结有该种子的单独Test结果。

## DeepSeek（可选）

沿用原来的显式开关和密钥环境变量；默认关闭。需要时，在启动后端前设置 ENABLE_DEEPSEEK=true 和 DEEPSEEK_API_KEY，在启动前端前设置 ENABLE_DEEPSEEK=true。不要把密钥写进文件；只在用户点击AI分析时调用外部服务，未测试付费请求。

## 停止和回退

两个服务终端分别Ctrl+C。不会停止旧服务，也不会改动旧默认模型。需退回旧系统时使用原启动指南，或直接在此网页选择旧模型回退。

本目录不执行训练或Test选型。新增自定义检测不改变已归档的Test成绩。

## 当前后台实例

本次交付已启动本地后端8002和网页8503。打开 http://127.0.0.1:8503 即可使用；重启电脑后需按前述脚本重新启动。

停止本次后台实例可在PowerShell运行：

```powershell
& "E:\yolo_project\evaluation_rebuild_20261031\post_test_delivery\web\stop_services.ps1"
```

停止脚本核对PID、创建时间、命令和工作目录，只停止本次记录的独立升级进程。后来手动重开的服务请在对应终端Ctrl+C。
