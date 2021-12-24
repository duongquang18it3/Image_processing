import scipy.misc
import numpy, math
import scipy.fftpack as fftim
from PIL import Image, ImageOps
import cv2 as cv

b = cv.imread('locmin.png')
#b = cv.cvtColor(a, cv.COLOR_BGR2GRAY)

width = b.shape[0]
height = b.shape[1]
F = 3
idTb = int((F*F)/2)+1
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

#tinh trung vi
for i in range(shapeNew, c.shape[1]-1):
	for j in range(shapeNew, c.shape[0]-1):
		mr = c[i][j][0]
		mg = c[i][j][1]
		mb = c[i][j][2]
		#---------------------------------------------------
		for h in range(i - shapeNew, i + shapeNew + 1):
			for k in range(j - shapeNew, j + shapeNew + 1):
				idr = int(c[h][k][0])
				mr = idr if (idr < mr) else mr

				idg = int(c[h][k][1])
				mg = idg if (idg < mg) else mg

				idb = int(c[h][k][2])
				mb = idb if (idb < mb) else mb

		#----------------------------------------------------
		b[i-shapeNew][j-shapeNew][0] = int(mr) if mr > 0 else 0
		b[i-shapeNew][j-shapeNew][1] = int(mg) if mg > 0 else 0
		b[i-shapeNew][j-shapeNew][2] = int(mb) if mb > 0 else 0

#luu anh
cv.imwrite('loc_min.jpg', b)

