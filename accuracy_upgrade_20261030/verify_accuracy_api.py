"""Exercise the optional accuracy API without starting or replacing any server."""
import os
import faulthandler
import sys
sys.stdout.reconfigure(line_buffering=True)
faulthandler.dump_traceback_later(30, repeat=False)
os.environ["ENABLE_DEEPSEEK"] = "false"
from pathlib import Path
from fastapi.testclient import TestClient
from accuracy_api import app


def main():
    import torch
    torch.set_num_threads(2)
    image = Path(r"D:\yolov8\NEU-DET\validation\images\crazing\crazing_241.jpg")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["status"] == "ok"
        assert health.json()["model"] == "clean_texture_accuracy_fusion"
        with image.open("rb") as file:
            response = client.post("/predict?conf=0.25&iou=0.3", files={"file": (image.name, file, "image/jpeg")})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["image"] == {"width": 200, "height": 200}
        assert data["count"] == len(data["detections"])
        for detection in data["detections"]:
            box = detection["box_xyxy"]
            assert 0 <= box["x1"] <= box["x2"] <= 200
            assert 0 <= box["y1"] <= box["y2"] <= 200
            assert detection["confidence"] >= 0.25
        invalid = client.post("/predict", files={"file": ("invalid.jpg", b"not an image", "image/jpeg")})
        assert invalid.status_code == 400, invalid.text
        bad_iou = client.post("/predict?iou=1.0", files={"file": ("invalid.jpg", b"invalid", "image/jpeg")})
        assert bad_iou.status_code == 422
        print("PASS health, actual-image prediction, coordinate bounds, threshold, corrupt upload, invalid IoU")
        print("Prediction count:", data["count"], "processing_ms:", data["processing_ms"])
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
