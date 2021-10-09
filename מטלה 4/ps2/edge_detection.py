import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# I used wikipedia to write this code
def edgeDetectionSobel(image, thresh = 0.7):
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    mag = np.zeros(image.shape)
    [rows,columns] = np.shape(image)
    output_image = np.zeros(shape=(rows, columns))

    for i in range(rows-2):
        for j in range(columns-2):
            S1 = np.sum(np.multiply(Gx, image[i:i + 3, j:j + 3]))
            S2 = np.sum(np.multiply(Gy, image[i:i + 3, j:j + 3]))
            mag[i+1,j+1] = np.sqrt(S1**2+S2**2)

    max = np.max(mag)
    thresh = thresh*max
    for i in range(1,rows):
        for j in range(1,columns):
            if mag[i,j] < thresh:
                output_image[i,j] = min(255, mag[i,j])

    output_image = output_image.astype(np.uint8)
    return output_image


def edgeDetectionCanny(image: np.ndarray, thrs_1: float, thrs_2: float):
    height, width = image.shape

    # step 1 - smooth with gaussian
    image = cv2.GaussianBlur(image, (5, 5), 1.4)

    # step 2 - compute partial derivatives
    lx = cv2.Sobel(np.float32(image), cv2.CV_64F, 1, 0, 3)
    ly = cv2.Sobel(np.float32(image), cv2.CV_64F, 0, 1, 3)

    # step 3 - compute magnitude and direction
    magnitude, direction = cv2.cartToPolar(lx, ly, angleInDegrees=True)

    for x in range(width):
        for y in range(height):

            #  step 4 - quantize the gradient directions to 4 sections
            gradient_directions = direction[y, x]
            gradient_directions = abs(gradient_directions - 180) if abs(gradient_directions) > 180 else abs(
                gradient_directions)

            # 0:[0,22.5)U(157.5,180]
            if gradient_directions < 22.5 or (157.5 < gradient_directions <= 180):
                first_neighbor_x, first_neighbor_y = x - 1, y
                second_neighbor_x, second_neighbor_y = x + 1, y

            # 45:[22.5,67.5)
            elif 22.5 <= gradient_directions < 67.5:
                first_neighbor_x, first_neighbor_y = x - 1, y - 1
                second_neighbor_x, second_neighbor_y = x + 1, y + 1

            # 90:[67.5,112.5)
            elif 67.5 <= gradient_directions < 112.5:
                first_neighbor_x, first_neighbor_y = x, y - 1
                second_neighbor_x, second_neighbor_y = x, y + 1

            # 135:[112.5,157.5)
            elif 112.5 <= gradient_directions < 157.5:
                first_neighbor_x, first_neighbor_y = x - 1, y + 1
                second_neighbor_x, second_neighbor_y = x + 1, y - 1

            # step 5 - preform NMS
            if width > first_neighbor_x >= 0 and height > first_neighbor_y >= 0:
                if magnitude[y, x] < magnitude[first_neighbor_y, first_neighbor_x]:
                    magnitude[y, x] = 0
                    continue

            if width > second_neighbor_x >= 0 and height > second_neighbor_y >= 0:
                if magnitude[y, x] < magnitude[second_neighbor_y, second_neighbor_x]:
                    magnitude[y, x] = 0

    # step 6.1 - define two threshholds
    mag_max = np.max(magnitude)
    T2 = mag_max * thrs_2
    T1 = mag_max * thrs_1

    # step 6.2
    for x in range(width):
        for y in range(height):

            gradient_magnitude = magnitude[y, x]

            # if its smaller than borh than its not edge
            if gradient_magnitude < T2:
                magnitude[y, x] = 0

            # if its greater than T2 than it is presumed to be an edge
            elif T1 > gradient_magnitude >= T2:
                magnitude[y, x] = 1

            # and if its greater than T1 than its an edge
            elif gradient_magnitude > T1:
                magnitude[y, x] = 2

    return magnitude
