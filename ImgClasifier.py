import cv2 as cv
import numpy as np
import os

path = 'D:\\DocumentosD\\Python\\Vision-artificial\\Imagenes\\Train' #Ruta donde se encuentran las imagenes de referencia
sift = cv.SIFT_create(nfeatures=3000) #función utilizada para crear un objeto sift.

#detect images 
images = [] #Lista para guardar las imagenes
classNames = [] #Lista para guardar el nombre de las imagenes
myList = os.listdir(path) #Lista para la lectura de los archivos de la ruta de referencia

print('Total classes detected: ',len(myList)) #Imprime la cantidad de archivos que se encuentran en la ruta

for cl in myList: #Para todos los archivos dentro de la ruta se guardan como imagen en escala de grises
    imgCur = cv.imread(f'{path}\\{cl}',0) #funcion para leer la imagen en escala de grises
    images.append(imgCur) #Se anexan a la lista de imagenes
    classNames.append(os.path.splitext(cl)[0]) #se anexa el nombre de la imagen a la lista de nombres
print(classNames) #se imprimen los nombres de las imagenes

def findDes(images): #Funcion para obtener los keypoint y descriptores de las imagenes de referencia
    desList=[] #Lista para guardar los descriptores de las referencias
    for img in images: #para cada imagen se obtiene el descriptor
        kp, des = sift.detectAndCompute(img, None) #funcion para obtener los keypoints y descriptores
        desList.append(des) #Se anexan los descriptores a la lista 
    return desList

def findId(img, desList, thres = 7):
    kp2, dp2 = sift.detectAndCompute(img, None)
    bf = cv.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            match = bf.knnMatch(des,dp2, k = 2)
            good = []
            for m,n in match:
                if m.distance < 0.47*n.distance:
                    good.append([m])
            matchList.append(len(good))
        
        print(matchList)
        
    except:
        pass
    
    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
            
    return finalVal
    
    


desList = findDes(images)
print(len(desList))

cap = cv.VideoCapture(0)



while True:
    ret, frame = cap.read()
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    id = findId(img, desList)
   
    if ret == True:
        if id != -1:
            cv.putText(frame, classNames[id],(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
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