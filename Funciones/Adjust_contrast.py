
from cProfile import label
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def rz(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

#leer imagen
img = rz(cv.imread('D:\\DocumentosD\\UTP\\brazo\\Train2\\Toilet_paper.jpeg',cv.IMREAD_COLOR), width= 400 )


#intensity
def contrast_img(img, intensity):
    if intensity >= 0 :
        intensity = np.absolute(intensity)
        intensity_max = np.ones(img.shape, dtype='uint8')*intensity
        bright_img = cv.multiply(img, intensity_max)
    else:
        intensity = np.absolute(intensity)
        intensity_max = np.ones(img.shape, dtype='uint8')*intensity
        bright_img = cv.divide(img, intensity_max)
    
    return bright_img

bimg = contrast_img(img, 2)
dimg = contrast_img(img, -2)
hist_img = cv.calcHist([cv.cvtColor(img, cv.COLOR_BGR2GRAY)], [0], None, [256], [0,256])
hist_bimg = cv.calcHist([cv.cvtColor(bimg, cv.COLOR_BGR2GRAY)], [0], None, [256], [0,256])
hist_dimg = cv.calcHist([cv.cvtColor(dimg, cv.COLOR_BGR2GRAY)], [0], None, [256], [0,256])

cv.imshow('result', np.hstack((img, bimg, dimg)))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

ax1.set_title('histogram')
ax1.set_xlabel('Bins')
ax1.set_ylabel('# of pixels')
ax1.plot(hist_img)
ax1.set_xlim([0,256])


ax2.set_title('histogram')
ax2.set_xlabel('Bins')
ax2.set_ylabel('# of pixels')
ax2.hist(cv.cvtColor(img, cv.COLOR_BGR2GRAY))



ax3.set_title('histogram')
ax3.set_xlabel('Bins')
ax3.set_ylabel('# of pixels')
ax3.plot(hist_dimg)
ax3.set_xlim([0,256])



plt.show()

cv.waitKey()
cv.destroyAllWindows()