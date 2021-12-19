import scipy.misc
import numpy, math
import scipy.fftpack as fftim
from PIL import Image, ImageOps
import cv2 as cv

image = Image.open('car.jpg')
#chuyen thanh anh xam
a = ImageOps.grayscale(image)

b = numpy.asarray(a)
#bieu dien FFT
c = fftim.fft2(b)
#thay đổi hình ảnh tần số Fourier
d = fftim.fftshift(c)
#intializing biến cho hàm tích chập
M = d.shape[0]
N = d.shape[1]
H = numpy.ones((M,N))
center1 = M/2
center2 = N/2
d_0 = 30.0 #bán kính cắt
#xác định hàm tích chập cho ILPF
for i in range(1,M):
	for j in range(1,N):
		r1 = (i-center1)**2+(j-center2)**2
		r = math.sqrt(r1)
		if r > d_0:
			H[i,j] = 0.0
#chuyenH thanh hinh anh
H = Image.fromarray(H)
#thực hiện phép chập
con = d * H
#tính toán độ lớn của FFT nghịch đảo
e = abs(fftim.ifft2(con))


cv.imwrite('a_ilowpass_output.jpg', e)
