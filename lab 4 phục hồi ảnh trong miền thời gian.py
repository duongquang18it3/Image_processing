from PIL import Image
import cv2

import numpy as np 
import matplotlib.pyplot as plt 

from PIL import Image, ImageFilter

from scipy.ndimage import maximum_filter, minimum_filter

import scipy.signal as signal



def contraharmonic_mean(img, size, Q):
    num = np.power(img, Q + 1)
    denom = np.power(img, Q)
    kernel = np.full(size, 1.0)
    result = cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)
    return result

def medianFilter(image):
    # triển khai ma trận 3x3 lọc trung vị
    processed_image = cv2.medianBlur(image, 3)
    # hiển thị ảnh
    cv2.imshow('Median Filter Processing', processed_image)
    # lưu kết quả
    cv2.imwrite('processed_image.png', processed_image)
    cv2.waitKey(0)

#Arithmetic mean filter:
def a_mean(img,kernel_size):
 
    G_mean_img = np.zeros(img.shape)
    #print(G_mean_img[0][0])
 
    #print(img)
    k = int((kernel_size-1)/2)
    #print(k)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i <k or i>(img.shape[0]-k-1) or j <k or j>(img.shape[1]-k-1):
                G_mean_img[i][j]=img[i][j]
            else:
                for n in range(kernel_size):
                    for m in range(kernel_size):
                        G_mean_img[i][j] +=np.float(1/(kernel_size*kernel_size)*img[i-k+n][j-k+m])
 
 
                #G_mean_img[i][j]=1/9*(img[i-1][j-1]+img[i-1][j]+img[i-1][j+1]+img[i][j-1]+img[i][j]+img[i][j+1]+img[i+1][j-1]+img[i+1][j]+img[i+1][j+1])
    G_mean_img = np.uint8(G_mean_img)
    return G_mean_img
 
 #Geometric mean filter:
def G_mean(img,kernel_size):
 
    G_mean_img = np.ones(img.shape)
    #print(G_mean_img[0][0])
 
    #print(img)
    k = int((kernel_size-1)/2)
    #print(k)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i <k or i>(img.shape[0]-k-1) or j <k or j>(img.shape[1]-k-1):
                G_mean_img[i][j]=img[i][j]
            else:
                for n in range(kernel_size):
                    for m in range(kernel_size):
                        G_mean_img[i][j] *=np.float(img[i-k+n][j-k+m])
                G_mean_img[i][j] = pow(G_mean_img[i][j],1/(kernel_size*kernel_size))
 
 
                #G_mean_img[i][j]=1/9*(img[i-1][j-1]+img[i-1][j]+img[i-1][j+1]+img[i][j-1]+img[i][j]+img[i][j+1]+img[i+1][j-1]+img[i+1][j]+img[i+1][j+1])
    G_mean_img = np.uint8(G_mean_img)
    return G_mean_img
 
 
 #Harmonic mean filter mean filter:
def H_mean(img,kernel_size):
 
    G_mean_img = np.zeros(img.shape)
    #print(G_mean_img[0][0])
 
    #print(img)
    k = int((kernel_size-1)/2)
    #print(k)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i <k or i>(img.shape[0]-k-1) or j <k or j>(img.shape[1]-k-1):
                G_mean_img[i][j]=img[i][j]
            else:
                for n in range(kernel_size):
                    for m in range(kernel_size):
                        if img[i-k+n][j-k+m] ==0:
                            G_mean_img[i][j] = 0
                            break
                        else:
                            G_mean_img[i][j] +=1/np.float(img[i-k+n][j-k+m])
                    else:
                        continue
                    break
                if G_mean_img[i][j]!=0:
                    G_mean_img[i][j] = (kernel_size*kernel_size)/G_mean_img[i][j]
 
 
                #G_mean_img[i][j]=1/9*(img[i-1][j-1]+img[i-1][j]+img[i-1][j+1]+img[i][j-1]+img[i][j]+img[i][j+1]+img[i+1][j-1]+img[i+1][j]+img[i+1][j+1])
    G_mean_img = np.uint8(G_mean_img)
    return G_mean_img
 
 
 #Inverse harmonic mean filter Mean filter:
def HT_mean(img,kernel_size,Q):
 
    G_mean_img = np.zeros(img.shape)
    #print(G_mean_img[0][0])
 
    #print(img)
    k = int((kernel_size-1)/2)
    #print(k)
 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i <k or i>(img.shape[0]-k-1) or j <k or j>(img.shape[1]-k-1):
                G_mean_img[i][j]=img[i][j]
            else:
                result_top = 0
                result_down = 0
                for n in range(kernel_size):
                    for m in range(kernel_size):
                        if Q>0:
                            result_top +=pow(np.float(img[i-k+n][j-k+m]),Q+1)
                            result_down +=pow(np.float(img[i-k+n][j-k+m]),Q)
                        else:
                            if img[i-k+n][j-k+m]==0:
                                G_mean_img[i][j] = 0
                                break
                            else:
                                result_top +=pow(np.float(img[i-k+n][j-k+m]),Q+1)
                                result_down +=pow(np.float(img[i-k+n][j-k+m]),Q)
                    else:
                        continue
                    break
 
                else:
                    if result_down !=0:
                        G_mean_img[i][j] = result_top/result_down
 
 
                #G_mean_img[i][j]=1/9*(img[i-1][j-1]+img[i-1][j]+img[i-1][j+1]+img[i][j-1]+img[i][j]+img[i][j+1]+img[i+1][j-1]+img[i+1][j]+img[i+1][j+1])
    G_mean_img = np.uint8(G_mean_img)
    return G_mean_img


def minFilter(im1): 
    # applying the min filter
    im2 = im1.filter(ImageFilter.MinFilter(size = 3))
    im2.show()

def maxFilter(im1):
    im2 = im1.filter(ImageFilter.MaxFilter(size = 3))
    im2.show()

def midpoint(img):
    maxf = maximum_filter(img, (3, 3))
    minf = minimum_filter(img, (3, 3))
    midpoint = (maxf + minf) / 2
    cv2.imshow(midpoint)

def guassFilter(img):
    img2 = cv2.imread('rose_salt_and_pepper.jpg')
    blur = cv2.GaussianBlur(img,(5,5),0)
    blur2 = cv2.GaussianBlur(img2,(5,5),0)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    blur_rgb2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2RGB)


    plt.subplot(221),plt.imshow(img_rgb),plt.title('Gauss Noise')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(blur_rgb),plt.title('Gauss Noise - Blurred')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(img_rgb2),plt.title('Salt&Pepper Noise')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(blur_rgb2),plt.title('Salt&Pepper Noise - Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    #image = '1.png'
    image = 'hoa.jpg'
    while True:
        print('Nhấn phím 1: Để thực hiện hồi phục ảnh bằng Contraharmonic.')
        print('Nhấn phím 2: Để thực hiện hồi phục ảnh bằng lọc trung vị.')
        print('Nhấn phím 3: Để thực hiện hồi phục ảnh bằng lọc trung bình số học.')
        print('Nhấn phím 4: Để thực hiện hồi phục ảnh bằng lọc Max.')
        print('Nhấn phím 5: Để thực hiện hồi phục ảnh bằng lọc Min.')
        print('Nhấn phím 6: Để thực hiện hồi phục ảnh bằng lọc Midpoint.')
        print('Nhấn phím 7: Để thực hiện hồi phục ảnh bằng lọc Guass.')
        
        print('Nhấn phím 0: Để kết thúc chương trình')
        print('')

        theInput = input("Nhấn phím tương ứng với công việc:\n")
        print('----------------------------------------------------------')
        print('')

        key = int(theInput)
        if key == 1:
            cv2_imshow(contraharmonic_mean(image, (3,3), 0.5))
        if key == 2:
            medianFilter(image)
        if key == 3:
            a_mean(image, 3)
        if key == 4:
            maxFilter(image)
        if key == 5:
            minFilter(image)
        if key == 6:
            midpoint(image)
        if key == 7:
            guassFilter(image)

        if key == 0:
            break

