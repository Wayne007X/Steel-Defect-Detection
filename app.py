import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. 网页基本设置
st.set_page_config(page_title="钢材缺陷检测系统", page_icon="🔍")
st.title("🔍 工业钢材表面缺陷检测系统")
st.write("请在下方上传一张钢材图片，AI 将自动检测裂纹、斑块等缺陷。")

# 2. 加载你的专属模型（路径已锁定为你训练好的 best.pt）
@st.cache_resource
def load_model():
    return YOLO(r"runs\neu_defect_train\weights\best.pt")

model = load_model()

# 3. 侧边栏：调节置信度阈值（为了让你能看到框，默认调低到 0.15）
st.sidebar.header("检测参数设置")
conf_threshold = st.sidebar.slider("置信度阈值 (越低越容易检测出缺陷)", 0.0, 1.0, 0.15, 0.05)

# 4. 网页上传图片组件
uploaded_file = st.file_uploader("请选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 把上传的图片转成 PIL 格式
    image = Image.open(uploaded_file)
    
    # 分成左右两列展示
    col1, col2 = st.columns(2)
    
    with col1:
        # 修复黄字警告：使用 width='stretch'
        st.image(image, caption="原始图片", width='stretch')
    
    with col2:
        with st.spinner('AI 正在分析中，请稍候...'):
            # 使用模型进行推理，传入滑块设置的阈值
            results = model.predict(image, conf=conf_threshold)
            
            # 获取画好框的图片结果
            res_plotted = results[0].plot()
            
            # 注意：YOLO 返回的是 BGR 格式，Streamlit 需要 RGB，所以需要转换一下
            res_rgb = res_plotted[:, :, ::-1]
            st.image(res_rgb, caption="检测结果", width='stretch')
            
    # 根据检测结果给出提示
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.success(f"检测完成！共发现 {len(boxes)} 个缺陷。")
    else:
        st.warning("检测完成！未发现明显缺陷。请尝试调低左侧的置信度阈值。")