import cv2 as cv
import numpy as np
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

    return cv.resize(image, dim, interpolation=inter)


sift = cv.SIFT_create(nfeatures=3000) #función utilizada para crear un objeto sift.

##detect images 
path = 'D:\\DocumentosD\\Python\\Vision-artificial\\Imagenes\\TrainF'
images = []
classNames = [] 

colors = [[],[],[],[]] #[gray, b, g, r]
not_colors = [[],[],[],[]]
color_name = ['Gray', 'Blue', 'Green', 'Red']
myList = os.listdir(path) 

cap = cv.VideoCapture('D:\\DocumentosD\\Python\\Vision-artificial\\Video\\frame\\Tijeras flash.mp4')
ret, frame = cap.read()   
h, w = frame.shape[:2]

print('Total classes detected: ',len(myList)) #Imprime la cantidad de archivos que se encuentran en la ruta

for cl in myList: 
    imgCur = cv.resize(cv.imread(f'{path}\\{cl}'), (w, h), interpolation=cv.INTER_AREA) 
    b, g, r = cv.split(imgCur)
    gy = cv.cvtColor(imgCur, cv.COLOR_BGR2GRAY)
    images.append(imgCur) 
    colors[0].append(gy)
    colors[1].append(b)
    colors[2].append(g)
    colors[3].append(r)
    not_colors[0].append(cv.bitwise_not(gy))
    not_colors[1].append(cv.bitwise_not(b))
    not_colors[2].append(cv.bitwise_not(g))
    not_colors[3].append(cv.bitwise_not(r))
    classNames.append(os.path.splitext(cl)[0]) 
print(classNames)



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
    
    
def bin(img, u):
    bw = [[],[],[],[]]
    for i in range(len(img[0])):  
        for j in range(len(img)):
            threshold, thresh = cv.threshold(img[j][i], u, 255,cv.THRESH_BINARY)
            bw[j].append(thresh)
    return bw

def blur_img(img, u):
    br = []
    for i in range(len(img)):  
        #b = cv.GaussianBlur(img[i], (u,u), cv.BORDER_DEFAULT)
        b = cv.medianBlur(img[i], u)
        br.append(b)
    return br

def join(img):
    br = []
    for i in range(len(img[0])):  
        aux = []
        for j in range(2):
            if j <= 1:
                aux.append(cv.bitwise_or(img[j][i], img[j+1][i]) )    
            else:
                aux.append(cv.bitwise_or(img[j][i], img[j+1][i]) )
        a = cv.bitwise_or(aux[0], aux[1]) 
        br.append(a)
        
    return br

def morph(img, opcion, size_kernel):
    mp = [[],[],[],[]]
    kernel = np.ones((size_kernel, size_kernel), np.uint8)
    for i in range(len(img[0])):  
        
        for j in range(len(img)):
            if opcion == 0:  
                a = cv.morphologyEx(img[j][i], cv.MORPH_OPEN, kernel)
            elif opcion == 1:
                a = cv.morphologyEx(img[j][i], cv.MORPH_CLOSE, kernel)
            elif opcion == 2:
                a = cv.erode(img[j][i], kernel, iterations=1)
            elif opcion == 3:
                a = cv.dilate(img[j][i], kernel, iterations=1)
            else:
                a = 0    
            mp[j].append(a)
       
    return mp

def morph_op(images, w, h):
    bw = bin(not_colors, u = 153) 
    mp = morph(bw, opcion=1, size_kernel=20)

    jn = join(mp)
    blur = blur_img(jn, 11)
    #show_sing(blur, classNames, '')

    new = []
    for i in range(len(images)):
        aux = cv.resize(cv.merge([blur[i],blur[i],blur[i]]), (w, h), interpolation=cv.INTER_AREA)
        new.append(cv.bitwise_and(images[i],aux))
        
    return new

def get_channels(imgCur):
    not_c = []
    b, g, r = cv.split(imgCur)
    gy = cv.cvtColor(imgCur, cv.COLOR_BGR2GRAY)
    images.append(imgCur) 
    not_c.append(cv.bitwise_not(gy))
    not_c.append(cv.bitwise_not(b))
    not_c.append(cv.bitwise_not(g))
    not_c.append(cv.bitwise_not(r))   
    return not_c

def show_sing(img, name, plus_name):
    for i in range(len(img)):
        show = cv.resize(img[i], (800,600), interpolation=cv.INTER_AREA )
        cv.imshow(name[i] + plus_name, show)

def morph_live(images, w, h):
    not_c = get_channels(images)
    bw = bin(not_c, u = 153) 
    mp = morph(bw, opcion=1, size_kernel=20)

    jn = join(mp)
    blur = blur_img(jn, 11)
    #show_sing(blur, classNames, '')

    aux = cv.resize(cv.merge([blur[0],blur[0],blur[0]]), (w, h), interpolation=cv.INTER_AREA)
    new = cv.bitwise_and(images,aux)
        
    return blur[0]

desList = morph_op(images,w, h)
desList = findDes(images)
print(len(desList))


while True:
    ret, frame = cap.read()
    frame = morph_live(frame, w, h)
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #id = findId(img, desList)
   
    if ret == True:
        # if id != -1:
        #     cv.putText(frame, classNames[id],(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
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