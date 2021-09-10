import cv2 as cv
import matplotlib.pyplot as plt

def dao_anh(img):
    return 255-img

def hien_thi_dao_anh():
    #định nghĩa vùng vẽ ảnh với phương thức figure
    fig = plt.figure(figsize=(16, 9))
    #tạo hai vùng vẽ ảnh
    ax1, ax2 = fig.subplots(1, 2)

    #su dung phuong thuc imread để mở ảnh
    img = cv.imread('daoanh.jpg',0)
    #hien thi anh trong vùng ax1
    ax1.imshow(img, cmap='gray')
    ax1.set_title("ẢNH GỐC")

    y = dao_anh(img)
    ax2.imshow(y, cmap='gray')
    ax2.set_title("ẢNH SAU KHI ĐẢO")
    plt.show()

if __name__ == '__main__':
  hien_thi_dao_anh()