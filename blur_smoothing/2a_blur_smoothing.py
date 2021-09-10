import os
import cv2
import numpy as np

INPUT = 'lena.jpg'
KERNEL_WIDTH = 9
KERNEL_HEIGHT = 9
SIGMA_X = 4
SIGMA_Y = 4

if not os.path.isfile(INPUT):
    raise Exception('File not found @ %s' % INPUT)

img = cv2.imread(INPUT)

blur_img = cv2.GaussianBlur(img, ksize=(
    KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)

numpy_horizontal_concat = np.concatenate((img, blur_img), axis=1)
cv2.imshow('Bai 2a', numpy_horizontal_concat)

cv2.waitKey()
