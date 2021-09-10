import cv2 as cv
import matplotlib.pyplot as plt

def Chuyen_doi_logarit(img, c):
    return float(c) * cv.log(1.0 + img)

def show_Chuyen_doi_logarit():
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('log.tif')
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Ảnh gốc")

    y = Chuyen_doi_logarit(img, 2)
    ax2.imshow(y, cmap='gray')
    ax2.set_title("Chuyển đổi bằng hàm Logarit")
    plt.show()

if __name__ == '__main__':
    show_Chuyen_doi_logarit()