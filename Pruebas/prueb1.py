import cv2 as cv
import numpy as np

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
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


path1 = 'D:\\DocumentosD\\UTP\\brazo\\Train\\'
path2 = 'D:\\DocumentosD\\UTP\\brazo\\Prueba\\'
img1 = cv.imread(path1+'r_pills.jpg')
img2 = cv.imread(path2+'a.jpg')

sift = cv.SIFT_create(nfeatures=3000)
kp1, dp1 = sift.detectAndCompute(img1, None)
kp2, dp2 = sift.detectAndCompute(img2, None)
img1 = cv.drawKeypoints(img1, kp1, None)
img2 = cv.drawKeypoints(img2, kp2, None)

bf = cv.BFMatcher()
match = bf.knnMatch(dp1,dp2, k = 2)

good = []
for m,n in match:
    print('m: ',m,' | n: ',n)
    if m.distance < 0.45*n.distance:
        good.append([m])
print(len(good))
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,match,None,flags=2)
img4 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

cv.imshow('Image',img1)
cv.imshow('Prueba',ResizeWithAspectRatio(img2, width=800))
#cv.imshow('Match',ResizeWithAspectRatio(img3,width=800))
cv.imshow('Match filtrado',ResizeWithAspectRatio(img4,width=800))
cv.waitKey(0)
cv.destroyAllWindows()