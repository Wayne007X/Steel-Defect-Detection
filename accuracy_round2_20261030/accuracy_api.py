"""Optional accuracy ONNX API, reusing the existing FastAPI/DeepSeek contract.

Run on a separate port; this module never edits the original API or checkpoints:
    python -m uvicorn accuracy_api:app --host 127.0.0.1 --port 8001 --workers 1
"""
import importlib.util
import os
from pathlib import Path
import sys

PROJECT = Path(os.environ.get("STEEL_PROJECT_ROOT", r"E:\yolo_project"))
LOCAL = Path(__file__).resolve().parent
TEXTURE_MODEL = LOCAL / "models/texture_ft_best.onnx"
if not TEXTURE_MODEL.is_file():
    TEXTURE_MODEL = LOCAL / "accuracy_experiments/training/mosaic000_texture_ft_seed42/weights/best.onnx"
MODEL_A = Path(os.environ.get("FUSION_MODEL_A_PATH", str(PROJECT / "runs/neu_defect_clean/weights/best.onnx")))
MODEL_B = Path(os.environ.get("FUSION_MODEL_B_PATH", str(TEXTURE_MODEL)))
for checkpoint in (MODEL_A, MODEL_B):
    if checkpoint.suffix.lower() != ".onnx" or not checkpoint.is_file():
        raise FileNotFoundError(f"Expected existing ONNX checkpoint: {checkpoint}")

os.environ["MODEL_PATH"] = str(MODEL_A)
os.environ["MODEL_NAME"] = "clean_texture_accuracy_fusion"
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(exist_ok=True, parents=True)
spec = importlib.util.spec_from_file_location("steel_original_api", PROJECT / "onnx_inference_api_pluggable_v4.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

from precision_detector import PrecisionDetector


def accuracy_model(*args, **kwargs):
    return PrecisionDetector([MODEL_A, MODEL_B], device="cpu")


base.YOLO = accuracy_model
app = base.app
app.title = "Steel Defect Accuracy Fusion API"
