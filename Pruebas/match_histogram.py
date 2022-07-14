# import packages
import matplotlib.pyplot as plt
from numpy import hstack
from skimage import exposure
from skimage.exposure import match_histograms
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

path = 'D:\\DocumentosD\\Python\\Vision-artificial\\Imagenes\\hist\\'

# reading reference image
img2 = cv.imread(path+'c1.jpg')
reference = img2
  
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    matched = match_histograms(frame, reference , channel_axis=-1)
    (h, w) = frame.shape[:2]
    reference = cv.resize(reference, (w, h), interpolation=cv.INTER_AREA)
    show = np.vstack( (rz(np.hstack((frame, reference)) , width=600 ), rz(matched, width=600)) )
    #prev = frame
    if ret == True:
        cv.imshow('frame', show)
        # cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('s'):
            break

cap.release()
cv.destroyAllWindows()
  