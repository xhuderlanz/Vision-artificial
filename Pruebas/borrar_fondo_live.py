import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

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



cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    (hframe, wframe) = frame.shape[:2]
    #prev = np.zeros((hframe, wframe, 3), dtype='uint8')
    prev = frame
    if ret == True:
        if cv.waitKey(1):
            break


while True:
    ret, frame = cap.read()
    dif = cv.absdiff(frame, prev)
    #prev = frame
    if ret == True:
        view = rz(np.hstack((frame, dif)),width=800)
        cv.imshow('frame', view)
        
        if cv.waitKey(1) & 0xFF == ord('s'):
            break

cap.release()
cv.destroyAllWindows()