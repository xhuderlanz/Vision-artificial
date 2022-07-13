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

gray = []
bw = []
i=1

fig1 = plt.figure("Filtro")
fig1.subplots_adjust(hspace=0.5, wspace=0.5)

for img in images:
    print('i: ',i)
    #gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gry
    threshold, thresh = cv.threshold(gry, 100, 255,cv.THRESH_BINARY)
    bw.append(thresh)
    gray_hist = cv.calcHist([gry], [0], None, [256], [0,256])
    gray.append(gry)
    
    
    ax = fig1.add_subplot(1, 2, i)
    ax.plot(gray_hist)
    ax.set_title('Grayscale histogram')
    ax.set_xlabel('Bins')
    ax.set_ylabel('# of pixels')
    ax.grid(color='gray', linestyle='dashed', linewidth=1, alpha=0.4)
    # Pintar los ejes pasando por (0,0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlim([0,256])
    i+=1
    #threshold, bw = cv.threshold(gray, 150, 255,cv.THRESH_BINARY)
cv.imshow('imagenes gray', np.hstack(gray))
cv.imshow('imagenes bw', np.hstack(bw))
plt.show()
cv.waitKey()
cv.destroyAllWindows()

# img =cv.bitwise_xor(images[0], images[1])

# cv.imshow('Diferencia', np.hstack((images[0], images[1], img)))
# cv.waitKey()
# cv.destroyAllWindows()