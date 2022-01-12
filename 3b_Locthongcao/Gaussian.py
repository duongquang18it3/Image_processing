import scipy.misc
import numpy, math
import scipy.fftpack as fftim
from PIL import Image, ImageOps
import cv2 as cv
a = Image.open('car.jpg').convert('L')
b = numpy.asarray(a)
c = fftim.fft2(b)
d = fftim.fftshift(c)
M = d.shape[0]
N = d.shape[1]
H = numpy.ones((M,N))
center1 = M/2
center2 = N/2
d_0 = 30.0
t1 = 2*d_0
for i in range(1,M):
        for j in range(1,N):
                r1 = (i-center1)**2+(j-center2)**2
                r = math.sqrt(r1)
                if 0 < r < d_0:
                        H[i,j] = 1 - math.exp(-r**2/t1**2)
H = Image.fromarray(H)
con = d * H
e = abs(fftim.ifft2(con))

cv.imwrite('gghightpass_output.jpg', e)
