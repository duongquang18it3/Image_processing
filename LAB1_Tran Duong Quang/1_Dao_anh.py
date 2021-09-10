import cv2 as cv
import matplotlib.pyplot as plt

def dao_anh(img):
    return 255-img

def show_dao_anh():
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('daoanh.tif',0)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("ảnh gốc")

    y = dao_anh(img)
    ax2.imshow(y, cmap='gray')
    ax2.set_title("ảnh đảo")
    plt.show()

if __name__ == '__main__':
    show_dao_anh()