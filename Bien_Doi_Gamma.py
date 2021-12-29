import cv2 as cv
import matplotlib.pyplot as plt

def Bien_Doi_Gamma(img, gamma, c):
    return float(c) * pow(img, float(gamma))

def hien_thi_gamma():
    fig = plt.figure(figsize=(16, 9))
    #2hang2cot
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)

    img = cv.imread('car.png',0)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("ẢNH GỐC")

    y1 = Bien_Doi_Gamma(img, 3.0, 1.0)
    ax2.imshow(y1, cmap='gray')
    ax2.set_title("GAMMA=3")

    y2 = Bien_Doi_Gamma(img, 4.0, 1.0)
    ax3.imshow(y2, cmap='gray')
    ax3.set_title("GAMMA=4")

    y3 = Bien_Doi_Gamma(img, 5.0, 1.0)
    ax4.imshow(y3, cmap='gray')
    ax4.set_title("GAMMA=5")
    plt.show()

if __name__ == '__main__':
    hien_thi_gamma()