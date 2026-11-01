"""Compare vectorized production fusion to the independent earlier implementation."""
import time
import numpy as np
from optimize_neu import OUT, dump
from compare_predictions import load, nms, fuse as original_fuse
from compare_adaptive_nms import adaptive_nms
from precision_detector import fuse


def main():
    results=[]
    for suffix in ("","_tta"):
        for mode,transform in (("native",lambda r:r),("standard",lambda r:nms(r,.3)),("adaptive",adaptive_nms)):
            pairs=[(transform(a),transform(b)) for a,b in zip(load("clean"+suffix),load("texture_ft"+suffix))]
            times=[]
            outputs=[]
            for function in (original_fuse,fuse):
                start=time.perf_counter()
                output=[function(a,b) for a,b in pairs]
                times.append(1000*(time.perf_counter()-start)/len(pairs))
                outputs.append(output)
            for before,after in zip(*outputs):
                for key in ("boxes","conf","cls"):
                    assert np.array_equal(before[key],after[key]),(suffix,mode,before["path"],key)
            row=dict(tta=bool(suffix),mode=mode,images=len(pairs),bitwise_identical=True,
                     original_fusion_ms_per_image=times[0],vectorized_fusion_ms_per_image=times[1])
            results.append(row);print(row,flush=True)
    dump(OUT / "round2/fusion_vectorization_verification.json",results)


if __name__=="__main__":
    main()
