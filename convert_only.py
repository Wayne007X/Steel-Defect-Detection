import os
import xml.etree.ElementTree as ET

# 严格按照你提供的真实路径
BASE_PATH = r"D:\yolov8\NEU-DET"
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def convert_xml_to_yolo(xml_file, output_label_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    if size is None: return
    w, h = int(size.find('width').text), int(size.find('height').text)
    
    lines = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text.strip()
        if cls_name not in CLASSES: continue
        cls_id = CLASSES.index(cls_name)
        box = obj.find('bndbox')
        xmin, xmax = float(box.find('xmin').text), float(box.find('xmax').text)
        ymin, ymax = float(box.find('ymin').text), float(box.find('ymax').text)
        
        x_center, y_center = (xmin + xmax) / 2.0 / w, (ymin + ymax) / 2.0 / h
        width, height = (xmax - xmin) / w, (ymax - ymin) / h
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(lines))

def process_dataset(subset):
    print(f"========== 开始处理 {subset} 集 ==========")
    ann_dir = os.path.join(BASE_PATH, subset, "annotations")
    
    if not os.path.exists(ann_dir):
        print(f"❌ 找不到 {ann_dir}，请检查你的路径！")
        return

    xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
    print(f"共找到 {len(xml_files)} 个 XML 文件，开始转换...")

    count = 0
    for xml_file in xml_files:
        # 提取类别名（例如 crazing_1.xml -> crazing）
        cls_name = xml_file.split('_')[0]
        # 对应的标签存放在 labels/类别名/ 下
        lbl_dir = os.path.join(BASE_PATH, subset, "labels", cls_name)
        
        xml_path = os.path.join(ann_dir, xml_file)
        txt_path = os.path.join(lbl_dir, xml_file.replace('.xml', '.txt'))
        
        try:
            convert_xml_to_yolo(xml_path, txt_path)
            count += 1
        except Exception as e:
            print(f"失败: {xml_file}, 错误: {e}")
            
    print(f"✅ {subset} 集处理完成，共生成 {count} 个标签文件。\n")

if __name__ == '__main__':
    process_dataset('train')
    process_dataset('validation')
    print("🎉 标签转换完毕！图片和文件夹结构完全未动。")