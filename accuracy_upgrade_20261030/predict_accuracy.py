"""Local precision inference, including optional CUDA TTA and saved annotated images."""
import argparse
import json
from pathlib import Path
import time
import torch
from precision_detector import PrecisionDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--project", type=Path, default=Path(r"E:\yolo_project"))
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=Path("accuracy_predictions"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    suffix = ".pt" if args.tta or args.device != "cpu" else ".onnx"
    weights = [args.project / "runs" / f"neu_defect_{n}" / "weights" / ("best"+suffix)
               for n in ("clean", "mosaic000")]
    local = Path(__file__).resolve().parent
    texture = local / "models" / ("texture_ft_best"+suffix)
    if not texture.is_file():
        texture = local / "accuracy_experiments/training/mosaic000_texture_ft_seed42/weights" / ("best"+suffix)
    weights[1] = texture
    detector = PrecisionDetector(weights, device=args.device, augment=args.tta)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result = detector.predict(args.source, conf=args.conf, iou=args.iou)[0]
    latency = (time.perf_counter()-start)*1000
    result.save(filename=str(args.output / args.source.name))
    payload = dict(source=str(args.source), models=list(map(str, weights)), tta=args.tta,
                   conf=args.conf, iou=args.iou, processing_ms=latency,
                   detections=result.summary())
    (args.output / (args.source.stem+".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
