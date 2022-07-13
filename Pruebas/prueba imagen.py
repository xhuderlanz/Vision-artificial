#D:\DocumentosD\UTP\brazo\prueba.jpg

import numpy as np
import cv2  
import matplotlib.pyplot as plt

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
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

    return cv2.resize(image, dim, interpolation=inter)




path = 'D:\\DocumentosD\\UTP\\brazo\\prueba.jpg'
img = cv2.imread(path,)

resize = ResizeWithAspectRatio(img, width=960)

cv2.imshow('resize', resize)
cv2.waitKey(0)


# cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# im = cv2.imread(path)                    # Read image
# imS = cv2.resize(im, (960, 540))                # Resize image
# cv2.imshow("output", imS)                       # Show image
# cv2.waitKey(0)   


