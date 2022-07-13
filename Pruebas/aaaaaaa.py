import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

path = 'D:\\DocumentosD\\Python\\Vision-artificial\\Imagenes\\Train' #Ruta donde se encuentran las imagenes de referencia
sift = cv.SIFT_create(nfeatures=1000)

#detect images
images = []
classNames = []
myList = os.listdir(path)

#print('Total classes detected: ', len(myList))



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


for cl in myList:
    imgCur = cv.imread(f'{path}\\{cl}')
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def subplot(images):
    lenImg = int(len(images))
    print('Total classes detected: ', len(images)) 
    y, x = 1, 1
    blank = np.zeros((500,500,3), dtype='uint8')
    img = blank
    
    if lenImg < 4:
        x , y = lenImg, 1 
        
        print("op 1")
    if lenImg == 4:
        x, y = 2,2
    if lenImg > 4 and lenImg < 9:
        x , y = lenImg, 2 
        print("op 2")
    img=rz(img, width = 800)
    return x, y,img
    


def findDes(images):
    desList=[]
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        desList.append(des)
    return desList

def findId(img, desList, thres = 30):
    kp2, des2 = sift.detectAndCompute(img, None)
    bf = cv.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            match = bf.knnMatch(des,des2, k = 2)
            good = []
            for m,n in match:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
        print(matchList)
    except:
        pass
    
    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
            
    return finalVal
    
# x , y , img = subplot(images)
# print(' x: ',x,' | y: ',y) 
i = 1


plt.show()

  

cv.waitKey()
cv.destroyAllWindows()

# desList = findDes(images)
# print(len(desList))

# cap = cv.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     id = findId(img, desList)
   
#     if ret == True:
        
#         cv.imshow('frame', frame)
#          if id != -1:
#             cv.putText(frame, classNames[id],(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
        
#         if cv.waitKey(1) & 0xFF == ord('s'):
#             break

# cap.release()
# cv.destroyAllWindows()