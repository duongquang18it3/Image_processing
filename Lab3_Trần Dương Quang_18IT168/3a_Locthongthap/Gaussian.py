import scipy.misc
import numpy, math
import scipy.fftpack as fftim
from PIL import Image, ImageOps
import cv2 as cv
# opening the image and converting it to grayscale
a = Image.open('car.jpg').convert('L')
# a is converted to an ndarray
b = numpy.asarray(a)
# performing FFT
c = fftim.fft2(b)
# shifting the Fourier frequency image126 Image Processing and Acquisition using Python
d = fftim.fftshift(c)
# intializing variables for convolution function
M = d.shape[0]
N = d.shape[1]
# H is defined and
# values in H are initialized to 1
H = numpy.ones((M,N))
center1 = M/2
center2 = N/2
d_0 = 30.0 # cut-off radius
t1 = 2*d_0
# defining the convolution function for GLPF
for i in range(1,M):
        for j in range(1,N):
                r1 = (i-center1)**2+(j-center2)**2
# euclidean distance from
# origin is computed
                r = math.sqrt(r1)
# using cut-off radius to
# eliminate high frequency
                if r > d_0:
                        H[i,j] = math.exp(-r**2/t1**2)
# converting H to an image
# H = PIL.toimage(H)
H = Image.fromarray(H)
# performing the convolution
con = d * H
# computing the magnitude of the inverse FFT
e = abs(fftim.ifft2(con))
# e is converted from an ndarray to an imageFourier Transform 127

# saving the image as glowpass_output.png in
# Figures folder

cv.imwrite('glowpass_output.jpg', e)
