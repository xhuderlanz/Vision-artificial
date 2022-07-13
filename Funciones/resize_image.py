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

    return cv.resize(image, dim,  interpolation=inter)

#Leer imagen
img = rz(cv.imread('D:\\DocumentosD\\UTP\\brazo\\Train2\\Toilet_paper.jpeg',cv.IMREAD_COLOR), width= 400 )