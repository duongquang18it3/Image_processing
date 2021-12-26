import scipy.misc
import numpy, math
import scipy.fftpack as fftim
from PIL import Image, ImageOps
import cv2 as cv

b = cv.imread('midpoint.png')
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
		mir = c[i][j][0]
		mig = c[i][j][1]
		mib = c[i][j][2]

		mar = mag = mab = 0
		#---------------------------------------------------
		for h in range(i - shapeNew, i + shapeNew + 1):
			for k in range(j - shapeNew, j + shapeNew + 1):
				idr = int(c[h][k][0])
				mir = idr if (idr < mir) else mir
				mar = idr if (idr > mar) else mar

				idg = int(c[h][k][1])
				mig = idg if (idg < mig) else mig
				mag = idr if (idr > mag) else mag

				idb = int(c[h][k][2])
				mib = idb if (idb < mib) else mib
				mab = idr if (idr > mab) else mab

		#----------------------------------------------------
		s = (mir + mar) /2
		z = (mig + mag) /2
		x = (mib + mab) /2
		#----------------------------------------------------
		b[i-shapeNew][j-shapeNew][0] = int(s) if s > 0 else 0
		b[i-shapeNew][j-shapeNew][1] = int(z) if z > 0 else 0
		b[i-shapeNew][j-shapeNew][2] = int(x) if x > 0 else 0

#luu anh
cv.imwrite('loc_midpoint.jpg', b)

