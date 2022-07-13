
import cv2 as cv
import numpy as np

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
def bright_img(img, intensity):
    if intensity >= 0 :
        intensity = np.absolute(intensity)
        intensity_max = np.ones(img.shape, dtype='uint8')*intensity
        bright_img = cv.add(img, intensity_max)
    else:
        intensity = np.absolute(intensity)
        intensity_max = np.ones(img.shape, dtype='uint8')*intensity
        bright_img = cv.subtract(img, intensity_max)
    
    return bright_img

bimg = bright_img(img, 100)
dimg = bright_img(img, -100)

cv.imshow('result', np.hstack((img, bimg, dimg)))
cv.waitKey()
cv.destroyAllWindows()