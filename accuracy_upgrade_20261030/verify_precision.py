"""End-to-end full validation of deployable detector, plus warmed latency measurements."""
import gc
import argparse
import json
import time
import numpy as np
import torch
from optimize_neu import OUT, PROJECT, dump
from compare_predictions import load, evaluate, counts, summarize
from precision_detector import PrecisionDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texture", action="store_true")
    args = parser.parse_args()
    prefix = "final_" if args.texture else ""
    torch.set_num_threads(2)
    gt = load("clean")
    weights = [PROJECT / "runs" / f"neu_defect_{name}" / "weights" / "best.onnx"
               for name in ("clean", "mosaic000")]
    if args.texture:
        weights[1] = OUT / "training/mosaic000_texture_ft_seed42/weights/best.onnx"
    # The exact same detector used by the optional API wrapper; ONNX on CPU.
    detector = PrecisionDetector(weights)
    detector.predict(gt[0]["path"])
    records, latencies = [], []
    for j, target in enumerate(gt):
        start = time.perf_counter()
        _, record = detector.records(target["path"], iou=0.7)
        latencies.append(1000*(time.perf_counter()-start))
        records.append({**record, "gt_boxes": target["gt_boxes"], "gt_cls": target["gt_cls"]})
        if j % 60 == 59:
            print("ONNX CPU verified", j+1, flush=True)
    metrics = evaluate(records)
    np.save(OUT / (prefix+"deployable_onnx_predictions.npy"), np.array(records, dtype=object), allow_pickle=True)
    metrics.update(name="deployable_onnx_fusion", images=len(records),
                   fixed_020=summarize(counts(records, [0.2]*6)),
                   fixed_025=summarize(counts(records, [0.25]*6)),
                   latency_cpu_ms=dict(median=float(np.median(latencies)), p95=float(np.percentile(latencies,95)),
                                       mean=float(np.mean(latencies)), batch=1, ort_threads_per_model=2,
                                       includes="image decode, preprocessing, both ONNX models, both NMS and fusion"))
    reference_file = "texture_fusion_comparison.json" if args.texture else "candidate_comparison.json"
    reference_name = "clean_texture_nms070" if args.texture else "fusion"
    reference = json.loads((OUT / reference_file).read_text(encoding="utf-8"))
    reference = next(r for r in reference if r["name"] == reference_name)
    assert abs(metrics["mAP50"] - reference["mAP50"]) < 0.002, (metrics, reference)
    dump(OUT / (prefix+"deployable_onnx_metrics.json"), metrics)
    print("ONNX FULL-DATASET PARITY PASSED", json.dumps(metrics), flush=True)
    del detector
    gc.collect()
    # This pipeline additionally supports PyTorch TTA. Measure and validate it
    # separately; ONNX mode never silently pretends to perform TTA.
    weights = [PROJECT / "runs" / f"neu_defect_{name}" / "weights" / "best.pt"
               for name in ("clean", "mosaic000")]
    if args.texture:
        weights[1] = OUT / "training/mosaic000_texture_ft_seed42/weights/best.pt"
    detector = PrecisionDetector(weights, device="cuda:0", augment=True)
    for nms_iou in (0.7, 0.3):
        detector.predict(gt[0]["path"], iou=nms_iou)
        records, latencies = [], []
        for j, target in enumerate(gt):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _, record = detector.records(target["path"], iou=nms_iou)
            torch.cuda.synchronize()
            latencies.append(1000*(time.perf_counter()-start))
            records.append({**record, "gt_boxes": target["gt_boxes"], "gt_cls": target["gt_cls"]})
        metrics = evaluate(records)
        metrics.update(name=f"deployable_pt_tta_nms{nms_iou}", images=len(records),
                       fixed_020=summarize(counts(records, [0.2]*6)),
                       fixed_025=summarize(counts(records, [0.25]*6)),
                       latency_gpu_ms=dict(median=float(np.median(latencies)), p95=float(np.percentile(latencies,95)),
                                           concurrent_training=not args.texture, batch=1))
        tag = prefix+"deployable_pt_tta" + ("_nms030" if nms_iou == 0.3 else "")
        np.save(OUT / (tag + "_predictions.npy"), np.array(records, dtype=object), allow_pickle=True)
        dump(OUT / (tag + "_metrics.json"), metrics)
        print("PT TTA VERIFIED", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
