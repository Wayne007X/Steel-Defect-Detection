# -*- coding: utf-8 -*-
r"""Live smoke tests for the running FastAPI ONNX service.

This script does not load/re-export a model and does not modify project files.
Run from PowerShell:
  python .\verify_fastapi_live.py "D:\yolov8\NEU-DET\validation\images\crazing_253.jpg"
Optional live DeepSeek call (may consume API quota):
  python .\verify_fastapi_live.py "D:\yolov8\NEU-DET\validation\images\crazing_253.jpg" --test-analyze
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running FastAPI ONNX inference service.")
    parser.add_argument("image", help="Path to an existing test image")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--expected-model", default="mosaic000", help="Expected model label from /health")
    parser.add_argument("--test-analyze", action="store_true", help="Call DeepSeek through /analyze; may consume API quota")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    base_url = args.base_url.rstrip("/")
    if not image_path.is_file():
        print(f"FAIL | test image not found: {image_path}")
        return 2

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, bool(passed), detail))
        print(f"{'PASS' if passed else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))

    session = requests.Session()
    try:
        health_response = session.get(f"{base_url}/health", timeout=5)
    except requests.RequestException as exc:
        print(f"FAIL | cannot reach {base_url}/health | {exc}")
        print("Make sure the FastAPI terminal is still running, then retry.")
        return 2

    try:
        health = health_response.json()
    except ValueError:
        health = {}
    check("GET /health returns 200 and status=ok", health_response.status_code == 200 and health.get("status") == "ok", f"HTTP {health_response.status_code}; status={health.get('status')}")
    check("configured model label matches expectation", health.get("model") == args.expected_model, f"expected={args.expected_model}; actual={health.get('model')}")
    check("health exposes class names", isinstance(health.get("class_names"), dict) and len(health.get("class_names", {})) >= 1, f"class_count={len(health.get('class_names', {}))}")
    if health_response.status_code != 200 or health.get("status") != "ok":
        print("\nStopping because the backend is not ready.")
        return 1

    try:
        with Image.open(image_path) as source:
            oriented_size = ImageOps.exif_transpose(source).size
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        image_bytes = image_path.read_bytes()
    except Exception as exc:
        print(f"FAIL | cannot read test image | {exc}")
        return 2

    try:
        response = session.post(
            f"{base_url}/predict",
            params={"conf": 0.25, "iou": 0.70},
            files={"file": (image_path.name, image_bytes, content_type)},
            timeout=120,
        )
    except requests.RequestException as exc:
        print(f"FAIL | POST /predict request failed | {exc}")
        return 2

    try:
        prediction: dict[str, Any] = response.json()
    except ValueError:
        prediction = {}
    check("POST /predict returns 200", response.status_code == 200, f"HTTP {response.status_code}; body={response.text[:300]}")
    if response.status_code != 200:
        print("\nStopping because /predict did not return a successful response.")
        return 1

    required = {"model", "filename", "image", "settings", "count", "processing_ms", "coordinate_system", "detections"}
    check("/predict response follows the expected JSON contract", required.issubset(prediction), f"missing={sorted(required - set(prediction))}")
    detections = prediction.get("detections") if isinstance(prediction.get("detections"), list) else []
    image_info = prediction.get("image") if isinstance(prediction.get("image"), dict) else {}
    check("reported image dimensions match EXIF-oriented upload", (image_info.get("width"), image_info.get("height")) == oriented_size, f"API={image_info.get('width')}x{image_info.get('height')}; image={oriented_size[0]}x{oriented_size[1]}")
    check("count equals number of detection entries", prediction.get("count") == len(detections), f"count={prediction.get('count')}; entries={len(detections)}")

    names = health.get("class_names", {})
    boxes_valid = True
    classes_valid = True
    confidence_valid = True
    for item in detections:
        box = item.get("box_xyxy") if isinstance(item, dict) else None
        try:
            x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
            boxes_valid &= 0 <= x1 <= x2 <= float(image_info["width"]) and 0 <= y1 <= y2 <= float(image_info["height"])
            class_id = int(item["class_id"])
            classes_valid &= str(class_id) in names and item.get("class_name") == names[str(class_id)]
            score = float(item["confidence"])
            confidence_valid &= 0.0 <= score <= 1.0
        except (KeyError, TypeError, ValueError):
            boxes_valid = classes_valid = confidence_valid = False
    check("all bounding boxes are valid original-image xyxy coordinates", boxes_valid, f"checked={len(detections)}")
    check("class IDs and names agree with /health", classes_valid, f"checked={len(detections)}")
    check("all confidence values are between 0 and 1", confidence_valid, f"checked={len(detections)}")
    print(f"INFO | model={prediction.get('model')}; image={image_info.get('width')}x{image_info.get('height')}; detections={len(detections)}; processing_ms={prediction.get('processing_ms')}")
    for item in detections[:5]:
        print(f"INFO | {item.get('class_name')} conf={item.get('confidence')} box={item.get('box_xyxy')}")

    # Error-handling tests: empty, malformed, too-large uploads and invalid parameters.
    for label, payload, expected_status in (
        ("empty upload is rejected with HTTP 400", b"", 400),
        ("non-image upload is rejected with HTTP 400", b"this is not an image", 400),
        ("oversized upload is rejected with HTTP 413", b"0" * (MAX_UPLOAD_BYTES + 1), 413),
    ):
        try:
            r = session.post(f"{base_url}/predict", files={"file": ("test.bin", payload, "application/octet-stream")}, timeout=30)
            check(label, r.status_code == expected_status, f"expected={expected_status}; actual={r.status_code}; body={r.text[:160]}")
        except requests.RequestException as exc:
            check(label, False, str(exc))

    try:
        r = session.post(
            f"{base_url}/predict",
            params={"conf": 0, "iou": 0.70},
            files={"file": (image_path.name, image_bytes, content_type)},
            timeout=30,
        )
        check("invalid confidence threshold is rejected with HTTP 422", r.status_code == 422, f"actual={r.status_code}")
    except requests.RequestException as exc:
        check("invalid confidence threshold is rejected with HTTP 422", False, str(exc))

    if args.test_analyze:
        try:
            r = session.post(f"{base_url}/analyze", json=prediction, timeout=40)
            try:
                analysis = r.json()
            except ValueError:
                analysis = {}
            valid_shape = r.status_code == 200 and analysis.get("status") in ("ok", "unavailable") and isinstance(analysis.get("message"), str)
            check("POST /analyze returns a structured success/fallback response", valid_shape, f"HTTP {r.status_code}; status={analysis.get('status')}; message={analysis.get('message')}")
            # Confirm vision still responds after the optional AI call.
            r2 = session.post(
                f"{base_url}/predict",
                params={"conf": 0.25, "iou": 0.70},
                files={"file": (image_path.name, image_bytes, content_type)},
                timeout=120,
            )
            check("/predict remains available after /analyze", r2.status_code == 200, f"HTTP {r2.status_code}")
        except requests.RequestException as exc:
            check("POST /analyze returns a structured success/fallback response", False, str(exc))

    print("\n" + "=" * 68)
    passed = sum(1 for _, ok, _ in checks if ok)
    failed = len(checks) - passed
    print(f"LIVE SMOKE TEST SUMMARY: {passed}/{len(checks)} passed; {failed} failed")
    if not args.test_analyze:
        print("DeepSeek live call skipped to avoid unexpected API usage. Add --test-analyze to include it.")
    print("Note: this script tests the running API; UI interaction and switching MODEL_PATH require a separate manual/restart test.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
