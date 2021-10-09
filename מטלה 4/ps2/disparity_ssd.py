import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def disparity_ssd(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    # the size of the window we compare
    window = 11
    max_offset = 100

    # left image and right image should have the same shape
    assert (L.shape == R.shape)

    # choose L/R arbitrary since they have the same shape
    hight, width = L.shape

    min = math.floor(window / 2)
    max_w = width - math.floor(window / 2)
    max_h = hight - math.floor(window / 2)

    # initial
    res = np.zeros((hight, width))
    for x in range(min, max_w):
        for y in range(min, max_h):
            w1 = L[y - min: y + min, x - min: x + min]
            dis = float('inf')
            for i in range(max_offset):
                w2 = R[y - min: y + min, x - min - i: x + min - i]
                if w2.shape == w1.shape:
                    d = np.subtract(w1, w2)
                    d = np.power(d, 2)
                    best = np.sum(d)
                    if best < dis:
                        dis = best
                        res[y, x] = i

    return res
