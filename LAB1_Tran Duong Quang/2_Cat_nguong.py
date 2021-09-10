import cv2 as cv
import matplotlib.pyplot as plt

def cat_nguong(img, th):
    return img > th

def show_cat_nguong():
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('keodan_dau.tif',0)
    ax1.imshow(img,cmap='gray')
    ax1.set_title("ảnh gốc")

    y = cat_nguong(img, th=117)
    ax2.imshow(y, cmap='gray')
    ax2.set_title("ảnh cắt ngưỡng")
    plt.show()

if __name__ == '__main__':
    show_cat_nguong()