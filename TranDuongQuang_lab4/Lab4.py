import cv2
import numpy as np
from math import sqrt, exp
import matplotlib.pyplot  as plt
from PIL import Image, ImageFilter
from scipy.ndimage import maximum_filter, minimum_filter


#Kỹ thuật lọc không gian được sử dụng trực tiếp trên các pixel của ảnh.
# Mặt nạ thường được coi là được thêm vào kích thước để nó có một pixel trung tâm cụ thể.
# Mặt nạ này được di chuyển trên hình ảnh sao cho tâm của mặt nạ đi ngang qua tất cả các pixel hình ảnh.
#mặt nạ vùng lân cận 3X3, 5X5 hoặc 7X7 có thể được xem xét. Ví dụ về mặt nạ 3X3 được hiển thị bên dưới:
#f (x-1, y-1) f (x-1, y) f (x-1, y + 1)
#f (x, y-1) f (x, y) f (x, y + 1)
#f (x + 1, y-1) f (x + 1, y) f (x + 1, y + 1)
class space_domain:
    def Averaging_Filter(self,img_CV2):
        # Nhận số hàng và cột của hình ảnh
        m, n = img_CV2.shape

        # Phát triển mặt nạ bộ lọc trung bình (3, 3)
        mask = np.ones([3, 3], dtype=int)
        mask = mask / 9

        # Chuyển đổi mặt nạ 3X3 trên hình ảnh
        img_out = np.zeros([m, n])

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = img_CV2[i - 1, j - 1] * mask[0, 0] + img_CV2[i - 1, j] * mask[0, 1] + img_CV2[i - 1, j + 1] * mask[0, 2] + \
                       img_CV2[i, j - 1] * mask[1, 0] + img_CV2[i, j] * mask[1, 1] + img_CV2[i, j + 1] * mask[1, 2] + img_CV2[
                           i + 1, j - 1] * mask[2, 0] + img_CV2[i + 1, j] * mask[2, 1] + img_CV2[i + 1, j + 1] * mask[2, 2]

                img_out[i, j] = temp

        img_out = img_out.astype(np.uint8)
        return img_out

    #Nó còn được gọi là lọc phi tuyến. Nó được sử dụng để loại bỏ nhiễu hạt tiêu
    # Ở đây giá trị pixel được thay thế bằng giá trị trung bình của pixel lân cận.
    def Median_Filter(self,img_CV2):
        # Nhận số hàng và cột của hình ảnh
        m, n = img_CV2.shape

        # Duyệt qua hình ảnh. Đối với mỗi vùng 3X3,
        # tìm trung vị của các pixel và thay thế pixel trung tâm bằng trung vị
        img_out = np.zeros([m, n])

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = [img_CV2[i - 1, j - 1],
                        img_CV2[i - 1, j],
                        img_CV2[i - 1, j + 1],
                        img_CV2[i, j - 1],
                        img_CV2[i, j],
                        img_CV2[i, j + 1],
                        img_CV2[i + 1, j - 1],
                        img_CV2[i + 1, j],
                        img_CV2[i + 1, j + 1]]

                temp = sorted(temp)
                img_out[i, j] = temp[4]
        img_out = img_out.astype(np.uint8)
        return img_out

    def Min_Filter(self,img_PIL):
        img_out = img_PIL.filter(ImageFilter.MinFilter(size=3))
        return img_out
    def Max_Filter(self, img_PIL):
        img_out = img_PIL.filter(ImageFilter.MaxFilter(size=3))
        return img_out
    def convert_PIL_to_CV2(self,img):
        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image
    def Mid_Point_Filter(self, img):
        maxf = maximum_filter(img, (3, 3))
        minf = minimum_filter(img, (3, 3))
        img_out = (maxf + minf) / 2
        return img_out
    def Contraharmonic_Filter(self,img,size,Q):
        num = np.power(img, Q + 1)
        denom = np.power(img, Q)
        kernel = np.full(size, 1.0)
        img_out = cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)
        return img_out

class frequency_domain:
    def distance(point1, point2):
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    def Gaussian_Filter(self,imgShape, D0):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows / 2, cols / 2)
        for x in range(cols):
            for y in range(rows):
                base[y, x] = exp(((-frequency_domain.distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
        return base
    def Ideal_Filter(self, imgShape, D0):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows / 2, cols / 2)
        for x in range(cols):
            for y in range(rows):
                if frequency_domain.distance((y, x), center) < D0:
                    base[y, x] = 1
        return base
    def Butterworth_Filter(self, imgShape, D0, n):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows / 2, cols / 2)
        for x in range(cols):
            for y in range(rows):
                base[y, x] = 1 / (1 + (frequency_domain.distance((y, x), center) / D0) ** (2 * n))
        return base

if __name__ == '__main__':
    img_CV2 = cv2.imread(r'hoa.jpg', 0)
    original = np.fft.fft2(img_CV2)
    center = np.fft.fftshift(original)
    # img_PIL = Image.open(r'C:\Users\Administrator\Desktop\GiaoTrinh\Khoa-VoNgocAnh-18IT075-lab 4\unnamed.jpg')
    img_PIL = Image.open(r'hoa.jpg')
    spa_dom = space_domain()

    img_averaging_filter = spa_dom.Averaging_Filter(img_CV2)
    #cv2.imshow('Averaging Filter', img_averaging_filter)

    img_median_filter = spa_dom.Median_Filter(img_CV2)
    #cv2.imshow('Medium Filter', img_median_filter)

    img_min_filter_before_process = spa_dom.Min_Filter(img_PIL)
    #img_min_filter_after_process = spa_dom.convert_PIL_to_CV2(img_min_filter_before_process)
    #cv2.imshow('Min Filter', img_min_filter_after_process)

    img_max_filter_before_process = spa_dom.Max_Filter(img_PIL)
    #img_max_filter_after_process = spa_dom.convert_PIL_to_CV2(img_max_filter_before_process)
    #cv2.imshow('Max Filter', img_max_filter_after_process)

    img_midpoint_filter = spa_dom.Mid_Point_Filter(img_CV2)

    img_contraharmonic_filter = spa_dom.Contraharmonic_Filter(img_CV2,(3,3),0.5)

    plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
    plt.figure(1), plt.figure(1).suptitle("Space Domain", fontsize=20)
    plt.subplot(2, 4, 1), plt.imshow(img_CV2, "gray"), plt.title("Original Image CV2")
    plt.subplot(2, 4, 2), plt.imshow(img_PIL, "gray"), plt.title("Orginal Image PIL")
    plt.subplot(2, 4, 3), plt.imshow(img_averaging_filter, "gray"), plt.title("Avengaring Filter")
    plt.subplot(2, 4, 4), plt.imshow(img_median_filter, "gray"), plt.title("Median Filter")
    plt.subplot(2, 4, 5), plt.imshow(img_min_filter_before_process, "gray"), plt.title("Min Filter")
    plt.subplot(2, 4, 6), plt.imshow(img_max_filter_before_process, "gray"), plt.title("Max Filter")
    plt.subplot(2, 4, 7), plt.imshow(img_midpoint_filter, "gray"), plt.title("Midpoint Filter")
    plt.subplot(2, 4, 8), plt.imshow(img_contraharmonic_filter, "gray"), plt.title("Contrahamonic Filter")

    fre_dom = frequency_domain()
    gausian_lp = center*fre_dom.Gaussian_Filter(img_CV2.shape,50)
    lowpass1 = np.fft.ifftshift(gausian_lp)
    inverse_lowpass1 = np.fft.ifft2(lowpass1)
    img_out_gausian_filter = np.abs(inverse_lowpass1)

    butterworth_lp = center * fre_dom.Butterworth_Filter(img_CV2.shape, 50, 10)
    lowpass2 = np.fft.ifftshift(butterworth_lp)
    inverse_lowpass2 = np.fft.ifft2(lowpass2)
    img_out_butterworth = np.abs(inverse_lowpass2)

    ideal_lp = center * fre_dom.Ideal_Filter(img_CV2.shape,50)
    lowpass3 = np.fft.ifftshift(ideal_lp)
    inverse_lowpass3 = np.fft.ifft2(lowpass3)
    img_out_ideal = np.abs(inverse_lowpass3)

    plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
    plt.figure(2), plt.figure(2).suptitle("Frequency Domain", fontsize=20)
    plt.subplot(2, 2, 1), plt.imshow(img_CV2, "gray"), plt.title("Original Image")
    plt.subplot(2, 2, 2), plt.imshow(img_out_gausian_filter, "gray"), plt.title("Gaussian Filter")
    plt.subplot(2, 2, 3), plt.imshow(img_out_butterworth, "gray"), plt.title("Butterworth Filter")
    plt.subplot(2, 2, 4), plt.imshow(img_out_ideal, "gray"), plt.title("Ideal Filter")
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


