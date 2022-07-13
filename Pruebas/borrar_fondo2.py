import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


path = 'D:\\DocumentosD\\Python\\Vision-artificial\\Imagenes\\background' #Ruta donde se encuentran las imagenes de referencia

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

#detect images 
images = [] #Lista para guardar las imagenes
classNames = [] #Lista para guardar el nombre de las imagenes
myList = os.listdir(path) #Lista para la lectura de los archivos de la ruta de referencia

print('Total classes detected: ',len(myList)) #Imprime la cantidad de archivos que se encuentran en la ruta

for cl in myList: #Para todos los archivos dentro de la ruta se guardan como imagen en escala de grises

    imgCur = rz(cv.imread(f'{path}\\{cl}'),width=400) #funcion para leer la imagen en escala de grises
    images.append(imgCur) #Se anexan a la lista de imagenes
    classNames.append(os.path.splitext(cl)[0]) #se anexa el nombre de la imagen a la lista de nombres
print(classNames) #se imprimen los nombres de las imagenes

img1 = cv.subtract(images[0], images[1])
img2 = cv.absdiff(images[0], images[1])
h1 = np.hstack((images[0], images[1]))
h2 = np.hstack((img1, img2))
v = np.vstack((h1,h2))

cv.imshow('imagen',v)
cv.waitKey()
cv.destroyAllWindows()