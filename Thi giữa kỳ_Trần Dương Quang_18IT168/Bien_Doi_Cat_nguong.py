import cv2 as cv
import matplotlib.pyplot as plt

def cat_nguong(img, th):
    return img > th

def hien_thi_cat_nguong():
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('car.png',0)
    ax1.imshow(img,cmap='gray')
    ax1.set_title("ẢNH GỐC")

    y = cat_nguong(img, th=117)
    ax2.imshow(y, cmap='gray')
    ax2.set_title("ẢNH CẮT NGƯỠNG")
    plt.show()

if __name__ == '__main__':
    hien_thi_cat_nguong()