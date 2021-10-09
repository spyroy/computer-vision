# ps2
import os
import numpy as np
import cv2
from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr
import matplotlib.pyplot as plt
import math
from edge_detection import edgeDetectionSobel, edgeDetectionCanny


def ps2():
    L = cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pair0-L.png', 0) * (
            1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pair0-R.png', 0) * (1.0 / 255.0)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('pair0-L')
    ax[0].imshow(L, cmap='gray')
    ax[1].set_title('pair0-R')
    ax[1].imshow(R, cmap='gray');
    plt.clf()


    D_L = disparity_ssd(R, L)

    plt.imshow(D_L, cmap='gray')
    plt.show()
    plt.clf()
    # plt.savefig('input/ps2-1-a-1.png')
    # files.download("ps2-1-a-1.png")

    D_R = disparity_ssd(L, R)

    plt.imshow(D_R, cmap='gray')
    plt.show()
    plt.clf()
    # plt.savefig('ps2-1-a-2.png')
    # files.download("ps2-1-a-2.png")

    real_img_L = cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pair1-L.png', 0) * (
                         1.0 / 255.0)
    real_img_R = cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pair1-R.png', 0) * (
                         1.0 / 255.0)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('pair1-L')
    ax[0].imshow(real_img_L, cmap='gray')
    ax[1].set_title('pair1-R', cmap='gray')
    ax[1].imshow(real_img_R);

    D_real_L = disparity_ssd(real_img_L, real_img_R)

    plt.imshow(D_real_L, cmap='gray')
    plt.show()
    plt.clf()
    # plt.savefig('ps2-2-a-1.png')
    # files.download("ps2-2-a-1.png")

    D_real_R = disparity_ssd(real_img_R, real_img_L)

    plt.imshow(D_real_R, cmap='gray')
    plt.show()
    plt.clf()
    # plt.savefig('ps2-2-a-2.png')
    # files.download("ps2-2-a-2.png")

    N_real_L = disparity_ncorr(real_img_L, real_img_R)

    plt.imshow(N_real_L, cmap='gray')
    plt.show()
    plt.clf()
    # plt.savefig('ps2-4-a-1.png')
    # files.download("ps2-4-a-1.png")

    # it seems to work faster with the norm

    N_real_R = disparity_ncorr(real_img_R, real_img_L)

    plt.imshow(N_real_R, cmap='gray')
    plt.show()
    plt.clf()
    # plt.savefig('ps2-4-a-2.png')
    # files.download("ps2-4-a-2.png")

    # it seems to work faster with the norm

    real_img_L2 = cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pair2-L.png', 0) * (
                          1.0 / 255.0)
    real_img_R2 = cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pair2-R.png', 0) * (
                          1.0 / 255.0)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('pair1-L')
    ax[0].imshow(real_img_L2, cmap='gray')
    ax[1].set_title('pair1-R')
    ax[1].imshow(real_img_R2, cmap='gray');
    plt.show()
    plt.clf()

    N_real_L2 = disparity_ncorr(real_img_L2, real_img_R2)

    plt.imshow(N_real_L2, cmap='gray')
    plt.show()
    plt.clf()
    # plt.savefig('ps2-4-b-1.png')
    # files.download("ps2-4-b-1.png")

    N_real_R2 = disparity_ncorr(real_img_R2, real_img_L2)

    plt.imshow(N_real_R2, cmap='gray')
    plt.show()
    plt.clf()
    # plt.savefig('ps2-4-b-2.png')
    # files.download("ps2-4-b-2.png")

    N_real_L2 = disparity_ncorr(real_img_R2, real_img_R2)

    plt.imshow(N_real_L2, cmap='gray')
    plt.show()
    plt.clf()
    # plt.savefig('ps2-4-b-2.png')
    # files.download("ps2-4-b-2.png")

    # same image for left and right should give us
    # 0 at all pixels because it will find
    # the exact same window at the other image
    # that is why we get black image

    # left to right works fine, but from right to left does not work that well,
    # the difference between without norm and with norm is mainly in the runtime,
    # and we get much noisier pictures than the groundtruth.

    img = cv2.cvtColor(cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pic1.png'),
                       cv2.COLOR_BGR2GRAY)
    img = edgeDetectionSobel(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()
    plt.clf()
    # plt.savefig('pic1 output sobel.png')
    # files.download("pic1 output sobel.png")

    img = cv2.cvtColor(cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pic2.png'),
                       cv2.COLOR_BGR2GRAY)
    img = edgeDetectionSobel(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()
    plt.clf()
    # plt.savefig('pic2 output sobel.png')
    # files.download("pic2 output sobel.png")

    img = cv2.cvtColor(cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pic1.png'),
                       cv2.COLOR_BGR2GRAY)
    img = edgeDetectionCanny(img, 0.2, 0.09)
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()
    plt.clf()
    # plt.savefig('pic1 output canny.png')
    # files.download("pic1 output canny.png")

    img = cv2.cvtColor(cv2.imread('C:/Users/spyro/PycharmProjects/ps2/input/pic2.png'),
                       cv2.COLOR_BGR2GRAY)
    img = edgeDetectionCanny(img, 0.2, 0.09)
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()
    plt.clf()
    # plt.savefig('pic2 output canny.png')
    # files.download("pic2 output canny.png")

    # It seems that canny is more detailed, and find more edges than sobel,
    # and it takes only the important edges.
