import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def disparity_ncorr(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    window = 11
    max_offset = 100
    L = L.astype(np.float32)
    R = R.astype(np.float32)

    assert (L.shape == R.shape)

    # Initial
    min = math.floor(window / 2)
    hight, width = L.shape
    max_w = width - math.floor(window / 2)
    max_h = hight - math.floor(window / 2)

    # Accumulate Normalized correlations for each offset
    res = np.zeros((hight, width, max_offset + 1), np.float)
    for y in range(min, max_h):
        for x in range(min, max_w):
            fac = x - min - max_offset
            if fac < 0:
                w1 = L[y - min: y + min + 1, x - min: x + min + 1]
                w2 = R[y - min: y + min + 1, : x + min + 1]
                res[y, x, : fac] = np.fliplr(cv2.matchTemplate(w2, w1, cv2.TM_CCOEFF_NORMED))
            else:
                w1 = L[y - min: y + min + 1, x - min: x + min + 1]
                w2 = R[y - min: y + min + 1, fac: x + min + 1]
                res[y, x, :] = np.fliplr(cv2.matchTemplate(w2, w1, cv2.TM_CCOEFF_NORMED))
    # Disparity images
    factor = 1.0 / max_offset
    d = res.argmax(-1) * factor
    return d
