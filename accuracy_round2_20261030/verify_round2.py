"""Re-run all 360 images through the real optional class-specific NMS detector."""
import argparse
import hashlib
import time
import numpy as np
import torch
from optimize_neu import OUT, MODELS, dump
from compare_predictions import load, nms, evaluate, counts, summarize
from compare_adaptive_nms import adaptive_nms
from precision_detector import PrecisionDetector, fuse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--adaptive",action="store_true")
    parser.add_argument("--concurrent-training",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    tag=f"seed{args.seed}_"+("adaptive" if args.adaptive else "standard")
    weights=[MODELS["clean"],OUT / f"training/mosaic000_texture_ft_seed{args.seed}/weights/best.pt"]
    detector=PrecisionDetector(weights,device="cuda:0",augment=True,
                                class_nms=[.5,.3,.3,.3,.5,.3] if args.adaptive else None)
    targets=load("clean")
    detector.predict(targets[0]["path"],iou=.3)
    records,latencies=[],[]
    for j,target in enumerate(targets):
        torch.cuda.synchronize()
        start=time.perf_counter()
        _,record=detector.records(target["path"],iou=.3)
        torch.cuda.synchronize()
        latencies.append(1000*(time.perf_counter()-start))
        records.append({**record,"gt_boxes":target["gt_boxes"],"gt_cls":target["gt_cls"]})
        if j%60 ==59:
            print(tag,j+1,flush=True)
    metrics=evaluate(records)
    target_name="texture_ft_tta" if args.seed ==42 else f"texture_ft_seed{args.seed}_tta"
    transform=adaptive_nms if args.adaptive else lambda r:nms(r,.3)
    reference=[fuse(transform(a),transform(b)) for a,b in zip(load("clean_tta"),load(target_name))]
    reference_metrics=evaluate(reference)
    assert abs(metrics["mAP50"]-reference_metrics["mAP50"])<.002
    assert abs(metrics["mAP50_95"]-reference_metrics["mAP50_95"])<.002
    result=dict(tag=tag,weights=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in weights],
                images=len(records),**metrics,fixed_025=summarize(counts(records,[.25]*6)),
                fixed_025_per_class=counts(records,[.25]*6),fixed_020=summarize(counts(records,[.2]*6)),
                latency_ms=dict(median=float(np.median(latencies)),p95=float(np.percentile(latencies,95)),
                                device="RTX 3060 Laptop CUDA",batch=1,
                                concurrent_training=args.concurrent_training,
                                includes="decode, preprocessing, two PyTorch TTA models, NMS, class-specific NMS if selected, fusion"),
                class_nms=detector.class_nms,global_iou=None if args.adaptive else .3,
                reference_AP50=reference_metrics["mAP50"],passed_cached_parity=True)
    np.save(OUT / "round2" / (tag+"_deploy_predictions.npy"),np.array(records,dtype=object),allow_pickle=True)
    dump(OUT / "round2" / (tag+"_deploy_metrics.json"),result)
    print("FULL DEPLOYMENT VERIFICATION",tag,metrics["mAP50"],result["fixed_025"],flush=True)


if __name__ == "__main__":
    main()
