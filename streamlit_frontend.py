# -*- coding: utf-8 -*-
"""Streamlit UI for the pluggable FastAPI + ONNX + DeepSeek steel-defect system.

The frontend does NOT load PyTorch, Ultralytics, or ONNX directly. It calls:
  POST {FASTAPI_BASE_URL}/predict  (multipart image upload)
  POST {FASTAPI_BASE_URL}/analyze  (JSON returned by /predict)

Run from E:\\yolo_project:
  python -m streamlit run streamlit_frontend.py
"""
from __future__ import annotations

import hashlib
import io
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


DEFAULT_API_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000").strip().rstrip("/")
# Disabled by default; enable explicitly in the local frontend process.
ENABLE_DEEPSEEK = os.getenv(
    "ENABLE_DEEPSEEK", "false"
).strip().lower() in {"1", "true", "yes", "on"}
PREDICT_TIMEOUT_SECONDS = 120
ANALYZE_TIMEOUT_SECONDS = 35
HEALTH_TIMEOUT_SECONDS = 3
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # Keep consistent with the FastAPI upload limit.
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")

st.set_page_config(
    page_title="钢材表面缺陷检测系统",
    page_icon="🔍",
    layout="wide",
)


@st.cache_data(ttl=10, show_spinner=False)
def get_health(api_base_url: str) -> tuple[bool, dict[str, Any] | str]:
    """Check the FastAPI backend without loading a model in Streamlit."""
    try:
        response = requests.get(f"{api_base_url}/health", timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return True, data
    except requests.RequestException as exc:
        return False, f"无法连接 FastAPI：{exc}"
    except ValueError:
        return False, "FastAPI /health 返回的不是有效 JSON。"


def make_request_key(api_base_url: str, image_bytes: bytes, conf: float, iou: float) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    return f"{api_base_url}|{digest}|conf={conf:.3f}|iou={iou:.2f}"


def load_oriented_rgb(image_bytes: bytes) -> Image.Image:
    """Decode the upload and apply EXIF orientation like the backend does."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
        return image


def draw_detections(image: Image.Image, prediction: dict[str, Any]) -> Image.Image:
    """Draw API-provided original-image xyxy boxes; do not run inference here."""
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    width, height = output.size
    line_width = max(2, round(min(width, height) / 250))
    try:
        font = ImageFont.truetype("arial.ttf", max(12, round(min(width, height) / 45)))
    except OSError:
        font = ImageFont.load_default()

    palette = [
        (220, 50, 47),
        (38, 139, 210),
        (133, 153, 0),
        (211, 54, 130),
        (181, 137, 0),
        (42, 161, 152),
        (108, 92, 231),
        (203, 75, 22),
    ]

    for detection in prediction.get("detections", []):
        box = detection.get("box_xyxy") or detection.get("box_xxyy") or {}
        try:
            x1 = int(round(float(box["x1"])))
            y1 = int(round(float(box["y1"])))
            x2 = int(round(float(box["x2"])))
            y2 = int(round(float(box["y2"])))
        except (KeyError, TypeError, ValueError):
            continue

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        try:
            class_id = int(detection.get("class_id", 0))
        except (TypeError, ValueError):
            class_id = 0
        color = palette[class_id % len(palette)]
        class_name = str(detection.get("class_name", f"class_{class_id}"))
        try:
            confidence = float(detection.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        label = f"{class_name} {confidence:.3f}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_x = max(0, min(x1, width - text_w - 6))
        label_y = y1 - text_h - 8
        if label_y < 0:
            label_y = min(y1 + 1, max(0, height - text_h - 6))
        draw.rectangle(
            (label_x, label_y, min(width - 1, label_x + text_w + 6), min(height - 1, label_y + text_h + 6)),
            fill=color,
        )
        draw.text((label_x + 3, label_y + 2), label, fill=(255, 255, 255), font=font)

    return output


def response_error_message(response: requests.Response) -> str:
    """Return a short useful error without dumping a huge HTML response."""
    try:
        body = response.json()
        detail = body.get("detail", body)
        return f"HTTP {response.status_code}: {detail}"
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:1200]}"


# Session state keeps the vision result visible while the optional AI request runs.
for state_key, initial_value in (
    ("predict_result", None),
    ("predict_key", None),
    ("predict_image_bytes", None),
    ("analysis_result", None),
    ("analysis_key", None),
):
    if state_key not in st.session_state:
        st.session_state[state_key] = initial_value


st.title("🔍 工业钢材表面缺陷检测系统")
st.caption(
    "mosaic000 ONNX · FastAPI 推理 · 可选 DeepSeek 辅助分析"
    if ENABLE_DEEPSEEK else "mosaic000 ONNX · FastAPI 推理"
)
st.write("上传一张钢材表面图片，系统会通过 FastAPI 调用当前配置的 ONNX 模型。")
if ENABLE_DEEPSEEK:
    st.write("视觉检测与 AI 分析分开执行，检测结果以本地模型输出为准。")

with st.sidebar:
    st.header("连接与检测设置")
    api_base_url = st.text_input(
        "FastAPI 地址",
        value=DEFAULT_API_BASE_URL,
        help="默认是本机 8000 端口。若后端地址改变，在这里修改即可，无需改代码。",
    ).strip().rstrip("/")
    conf_threshold = st.slider(
        "置信度阈值",
        min_value=0.01,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="阈值越低，通常会显示更多候选检测框，也可能增加误检。",
    )
    nms_iou = st.slider(
        "NMS IoU 阈值",
        min_value=0.05,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="该参数传给后端 /predict，不在前端重新执行 NMS。",
    )
    st.divider()
    if api_base_url:
        backend_ok, health_info = get_health(api_base_url)
        if backend_ok and isinstance(health_info, dict) and health_info.get("status") == "ok":
            st.success("FastAPI 已连接")
            st.caption(f"模型：{health_info.get('model') or '后端已就绪'}")
            st.caption(f"设备：{health_info.get('device') or '由后端决定'}")
        elif backend_ok:
            st.warning(f"FastAPI 已响应，但模型未就绪：{health_info}")
        else:
            st.error(str(health_info))
            st.caption("请先在另一个终端启动 FastAPI，再点击检测。")

uploaded_file = st.file_uploader(
    "上传钢材表面图片",
    type=list(ALLOWED_EXTENSIONS),
    help="支持 JPG、PNG、WEBP、BMP、TIFF；单张图片最大 10 MiB。",
)

uploaded_bytes: bytes | None = uploaded_file.getvalue() if uploaded_file is not None else None
current_key: str | None = None
current_image: Image.Image | None = None

if uploaded_bytes is not None:
    if len(uploaded_bytes) > MAX_UPLOAD_BYTES:
        st.error("图片超过 10 MiB，请压缩图片后重新上传。")
        uploaded_bytes = None
    else:
        try:
            current_image = load_oriented_rgb(uploaded_bytes)
            current_key = make_request_key(api_base_url, uploaded_bytes, conf_threshold, nms_iou)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            st.error(f"无法读取该图片，请换一张有效图片。详情：{exc}")
            uploaded_bytes = None
            current_image = None

if current_image is not None:
    st.image(current_image, caption=f"原始图片 · {current_image.width} × {current_image.height}", width=420)

predict_clicked = st.button(
    "开始视觉检测",
    type="primary",
    disabled=(uploaded_bytes is None or not api_base_url),
    use_container_width=True,
)

if predict_clicked and uploaded_bytes is not None and current_key is not None:
    try:
        with st.spinner("正在调用 FastAPI /predict…"):
            response = requests.post(
                f"{api_base_url}/predict",
                params={"conf": conf_threshold, "iou": nms_iou},
                files={
                    "file": (
                        uploaded_file.name if uploaded_file is not None else "upload.png",
                        uploaded_bytes,
                        uploaded_file.type if uploaded_file is not None and uploaded_file.type else "application/octet-stream",
                    )
                },
                timeout=PREDICT_TIMEOUT_SECONDS,
            )
            if not response.ok:
                raise RuntimeError(response_error_message(response))
            prediction = response.json()
            if not isinstance(prediction, dict) or not isinstance(prediction.get("detections"), list):
                raise RuntimeError("后端响应格式不符合预期，缺少 detections 列表。")

        st.session_state["predict_result"] = prediction
        st.session_state["predict_key"] = current_key
        st.session_state["predict_image_bytes"] = uploaded_bytes
        st.session_state["analysis_result"] = None
        st.session_state["analysis_key"] = None
        st.success("视觉检测完成。")
    except requests.Timeout:
        st.error("调用 /predict 超时。请确认后端仍在运行后重试。")
    except requests.RequestException as exc:
        st.error(f"无法连接 FastAPI /predict：{exc}")
    except (ValueError, RuntimeError) as exc:
        st.error(f"视觉检测失败：{exc}")

result_is_current = (
    current_key is not None
    and st.session_state["predict_result"] is not None
    and st.session_state["predict_key"] == current_key
)

if result_is_current:
    prediction = st.session_state["predict_result"]
    detections = prediction.get("detections", [])

    st.divider()
    st.header("视觉检测结果")
    metric_cols = st.columns(4)
    metric_cols[0].metric("检测框数量", int(prediction.get("count", len(detections))))
    metric_cols[1].metric("推理耗时", f"{float(prediction.get('processing_ms', 0.0)):.1f} ms")
    metric_cols[2].metric("模型", str(prediction.get("model", "unknown")))
    image_info = prediction.get("image") or {}
    metric_cols[3].metric(
        "图片尺寸",
        f"{image_info.get('width', '?')} × {image_info.get('height', '?')}",
    )

    try:
        result_image = load_oriented_rgb(st.session_state["predict_image_bytes"])
        annotated_image = draw_detections(result_image, prediction)
        image_col, summary_col = st.columns([1.3, 1])
        with image_col:
            st.image(annotated_image, caption="检测框可视化（坐标来自 FastAPI）", width=360)
        with summary_col:
            st.subheader("检测概览")
            if not detections:
                st.info("当前阈值下没有检测到缺陷框。这不代表图像中绝对不存在缺陷。")
            else:
                class_counts: dict[str, int] = {}
                for item in detections:
                    name = str(item.get("class_name", "unknown"))
                    class_counts[name] = class_counts.get(name, 0) + 1
                for class_name, count in sorted(class_counts.items(), key=lambda pair: (-pair[1], pair[0])):
                    st.write(f"- **{class_name}**：{count} 个")
                st.caption(
                    f"置信度阈值：{prediction.get('settings', {}).get('conf', conf_threshold)} · "
                    f"NMS IoU：{prediction.get('settings', {}).get('iou', nms_iou)}"
                )
    except (UnidentifiedImageError, OSError, TypeError) as exc:
        st.warning(f"检测结果已返回，但无法绘制预览图：{exc}")

    with st.expander("查看完整检测框数据"):
        if detections:
            st.dataframe(detections, use_container_width=True, hide_index=True)
        else:
            st.write("没有检测框。")

    with st.expander("查看原始 /predict JSON"):
        st.json(prediction)

    if ENABLE_DEEPSEEK:
        st.divider()
        st.header("DeepSeek 辅助分析")
        st.caption("这是独立请求；只分析检测器返回的结构化数据，不会直接查看原始图片。")
        analyze_clicked = st.button("生成 AI 辅助分析", key="analyze_button", use_container_width=True)

        if analyze_clicked:
            try:
                with st.spinner("正在调用 FastAPI /analyze，等待 DeepSeek 返回…"):
                    response = requests.post(
                        f"{api_base_url}/analyze",
                        json=prediction,
                        timeout=ANALYZE_TIMEOUT_SECONDS,
                    )
                    if not response.ok:
                        raise RuntimeError(response_error_message(response))
                    analysis = response.json()
                st.session_state["analysis_result"] = analysis
                st.session_state["analysis_key"] = current_key
            except requests.Timeout:
                st.session_state["analysis_result"] = {
                    "status": "unavailable",
                    "message": "前端等待 AI 分析超时。视觉检测结果仍然有效，可以稍后重新请求分析。",
                }
                st.session_state["analysis_key"] = current_key
            except requests.RequestException as exc:
                st.session_state["analysis_result"] = {
                    "status": "unavailable",
                    "message": f"无法连接 /analyze：{exc}。视觉检测结果仍然有效。",
                }
                st.session_state["analysis_key"] = current_key
            except (ValueError, RuntimeError) as exc:
                st.session_state["analysis_result"] = {
                    "status": "unavailable",
                    "message": f"AI 分析请求失败：{exc}。视觉检测结果仍然有效。",
                }
                st.session_state["analysis_key"] = current_key

        if st.session_state["analysis_result"] is not None and st.session_state["analysis_key"] == current_key:
            analysis = st.session_state["analysis_result"]
            if analysis.get("status") == "ok" and analysis.get("analysis"):
                st.success(analysis.get("message", "AI 分析完成。"))
                st.markdown(analysis["analysis"])
                if analysis.get("latency_ms") is not None:
                    st.caption(f"AI 接口耗时：{float(analysis['latency_ms']):.0f} ms")
            else:
                st.warning(analysis.get("message", "AI 分析暂不可用；视觉检测结果仍然有效。"))

st.caption(
    "提示：模型路径在 FastAPI 后端通过 MODEL_PATH 配置；"
    + ("请勿在 Streamlit 前端重复加载模型或填写 DeepSeek API Key。"
       if ENABLE_DEEPSEEK else "Streamlit 前端仅负责上传图片和展示检测结果。")
)
