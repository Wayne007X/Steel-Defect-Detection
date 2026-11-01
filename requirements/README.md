# 环境版本与用途

- `frontend.txt`：引用根目录原requirements.txt，供原Streamlit前端使用；没有修改原云端依赖文件。
- `api.txt`：原开发机推理/API直接依赖的已安装版本，含PyTorch/Ultralytics/ONNX Runtime。
- `research.txt`：研究/导出环境，在API环境基础上增加ONNX。
- [版本与冻结环境记录](../docs/evidence/environment_versions.json)：当前观察值及训练时保存的实现哈希。

这些文件记录直接依赖，不是包含全部传递依赖的锁文件，也没有在全新虚拟环境中重新安装验收。PyTorch的`+cu121`构建需要匹配的wheel来源和CUDA硬件条件，不能承诺仅用默认pip源即可安装。

分组划分的scikit-learn 1.5.2等依赖继续通过原[隔离安装脚本](../evaluation_rebuild_20261031/setup_split_dependencies.ps1)安装到专属目录，不升级训练环境。该脚本沿用原开发机路径，跨机器应建立新协议版本后配置。

只查看报告和运行默认仓库证据核验器不需要这些GPU依赖，Python 3.10+标准库即可。完整本地核验才需要原环境、数据、权重与预测缓存。
