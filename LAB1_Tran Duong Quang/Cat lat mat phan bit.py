import numpy as np
import cv2
import math

def change_binary(img,lst):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            lst.append(np.binary_repr(img[x][y],width=8))

def turn_to_img(img,lis,bits):

    a=8-bits

    bit_img=np.array([int(i[a])*math.pow(2,bits-1) for i in lis],dtype=np.uint8).reshape(img.shape[0],img.shape[1])

    return bit_img

def img_mix_show(eight_bit_img,seven_bit_img,six_bit_img,five_bit_img,four_bit_img,three_bit_img,two_bit_img,one_bit_img):
    a1 = cv2.hconcat([eight_bit_img, seven_bit_img, six_bit_img, five_bit_img])
    a2 = cv2.hconcat([four_bit_img, three_bit_img, two_bit_img, one_bit_img])
    final_imag = cv2.vconcat([a1, a2])
    cv2.namedWindow('final', cv2.WINDOW_NORMAL)
    cv2.imshow('final', final_imag)
    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    Binarry_code = []
    change_binary(img, Binarry_code)
    eight_bit_img = turn_to_img(img, Binarry_code, 8)
    seven_bit_img = turn_to_img(img, Binarry_code, 7)
    six_bit_img = turn_to_img(img, Binarry_code, 6)
    five_bit_img = turn_to_img(img, Binarry_code, 5)
    four_bit_img = turn_to_img(img, Binarry_code, 4)
    three_bit_img = turn_to_img(img, Binarry_code, 3)
    two_bit_img = turn_to_img(img, Binarry_code, 2)
    one_bit_img = turn_to_img(img, Binarry_code, 1)

    img_mix_show(eight_bit_img, seven_bit_img, six_bit_img, five_bit_img, four_bit_img, three_bit_img, two_bit_img,
                 one_bit_img)