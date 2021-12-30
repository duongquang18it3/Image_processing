import cv2 as cv
import matplotlib.pyplot as plt

def Bien_Doi_Logarit(img, c):
    return float(c) * cv.log(1.0 + img)

def hien_thi_logarit():
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('logit.jpg')
    ax1.imshow(img, cmap='gray')
    ax1.set_title("ẢNH GỐC")

    y = Bien_Doi_Logarit(img, 5)
    ax2.imshow(y, cmap='gray')
    ax2.set_title("ẢNH SAU KHI CHUYỂN ĐỔI BẰNG LOGARIT")
    plt.show()

if __name__ == '__main__':
    hien_thi_logarit()