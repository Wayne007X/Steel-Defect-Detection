import os

# 你的绝对路径
BASE_PATH = r"D:\yolov8\NEU-DET"

# data.yaml 的内容，注意这里的路径格式
yaml_content = f"""path: {BASE_PATH}
train: train/images
val: validation/images
nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
"""

# 写入到数据集根目录下
yaml_path = os.path.join(BASE_PATH, "data.yaml")
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"🎉 成功！data.yaml 已经生成在: {yaml_path}")
print("请去该目录下看一眼，确认它已经存在。")