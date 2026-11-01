"""Image-local grayscale percentile normalization; no dataset-fitted statistics."""
import cv2
import numpy as np


def normalize_luminance(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    low,high=np.percentile(gray,[1,99])
    if high-low <1:
        normalized=gray.copy()
    else:
        normalized=np.clip((gray.astype(np.float32)-low)*255/(high-low),0,255).astype(np.uint8)
    return cv2.cvtColor(normalized,cv2.COLOR_GRAY2BGR)
