import cv2 as cv
import numpy as np
import os

path = 'D:\\DocumentosD\\Python\\Vision-artificial\\Imagenes\\Train' #Ruta donde se encuentran las imagenes de referencia
sift = cv.SIFT_create(nfeatures=3000)

#detect images
images = []
classNames = []
myList = os.listdir(path)

print('Total classes detected: ',len(myList))

for cl in myList:
    imgCur = cv.imread(f'{path}\\{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findDes(images):
    desList=[]
    kpList=[]
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        desList.append(des)
        kpList.append(kp)
    return desList, kpList

def findId(img, desList, thres = 8):
    kp2, dp2 = sift.detectAndCompute(img, None)
    bf = cv.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            match = bf.knnMatch(des,dp2, k = 2)
            good = []
            for m,n in match:
                if m.distance < 0.45*n.distance:
                    good.append([m])
            matchList.append(len(good))
        
        print(matchList)
        
    except:
        pass
    
    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
            
    return finalVal, kp2, good
    
    


desList, kpList = findDes(images)
print(len(desList))

cap = cv.VideoCapture(0)



while True:
    ret, frame = cap.read()
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    id, kp2, match = findId(img, desList)
    if ret == True:
        if id != -1:
            
            cv.putText(frame, classNames[id],(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
            frame = cv.drawMatchesKnn(images[id],kpList[id],frame,kp2,match,None,flags=2)

            
        cv.imshow('frame', frame)
        
        if cv.waitKey(1) & 0xFF == ord('s'):
            break

cap.release()
cv.destroyAllWindows()

'''
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


img1 = cv.drawKeypoints(img1, kp1, None)
img2 = cv.drawKeypoints(img2, kp2, None)

bf = cv.BFMatcher()
match = bf.knnMatch(dp1,dp2, k = 2)

good = []
for m,n in match:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

cv.imshow('Image',img1)
cv.imshow('Prueba',ResizeWithAspectRatio(img2, width=800))
cv.imshow('Match',ResizeWithAspectRatio(img3,width=800))
cv.waitKey(0)
cv.destroyAllWindows()
'''