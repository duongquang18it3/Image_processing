import scipy.misc
import numpy, math
import scipy.fftpack as fftim
from PIL import Image, ImageOps
import cv2 as cv

b = cv.imread('monic.jpg')
#b = cv.cvtColor(a, cv.COLOR_BGR2GRAY)

width = b.shape[0]
height = b.shape[1]
F = 3
shapeNew = int(F/2)
c = numpy.zeros((width + 2 * shapeNew, height + 2 * shapeNew, 3))


#nhan doi vung bien
c[0][0] = b[0][0]
c[0][width+1] = b[0][width-1]
c[height+1][0] = b[height-1][0]
c[height+1][width+1] = b[height-1][width-1]

c[0][1:width+1] = b[0][0:width]
c[height+1][1:width+1] = b[height-1][0:width]

for i in range(1,height+1):
	c[i][0] = b[i-1][0]
	c[i][width+1] = b[i-1][width-1]

for i in range(shapeNew, c.shape[1]-1):
	for j in range(shapeNew, c.shape[0]-1):
		c[i][j] = b[i-shapeNew][j-shapeNew]

#tinh trung binh
for i in range(shapeNew, c.shape[1]-1):
	for j in range(shapeNew, c.shape[0]-1):
		r = c[i-1][j-1][0]*c[i-1][j][0]*c[i-1][j+1][0]*c[i][j-1][0]*c[i][j][0]*c[i][j+1][0]*c[i+1][j-1][0]*c[i+1][j][0]*c[i+1][j+1][0]
		r = r**(1/9) 
		b[i-shapeNew][j-shapeNew][0] = int(r) if r > 0 else 0

		g = c[i-1][j-1][1]*c[i-1][j][1]*c[i-1][j+1][1]*c[i][j-1][1]*c[i][j][1]*c[i][j+1][1]*c[i+1][j-1][1]*c[i+1][j][1]*c[i+1][j+1][1]
		g = g**(1/9) 
		b[i-shapeNew][j-shapeNew][1] = int(g) if g > 0 else 0

		p = c[i-1][j-1][2]*c[i-1][j][2]*c[i-1][j+1][2]*c[i][j-1][2]*c[i][j][2]*c[i][j+1][2]*c[i+1][j-1][2]*c[i+1][j][2]*c[i+1][j+1][2]
		p = p**(1/9) 
		b[i-shapeNew][j-shapeNew][2] = int(p) if p > 0 else 0

#luu anh
cv.imwrite('mean_fillter.jpg', b)

