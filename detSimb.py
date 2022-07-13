import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt

def detect(img, recorte):
    #algoritmo
    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    recorte = cv.cvtColor(recorte, cv.COLOR_BGR2GRAY)
    w = int(recorte.shape[1])
    h = int(recorte.shape[0]) # Extraemos el ancho y el alto del recorte del objeto)
    deteccion  = cv.matchTemplate(gris, recorte, cv.TM_CCOEFF_NORMED)  # Realizamos la deteccion por patrones
    print('detecion: ',deteccion)
    umbral = 0.75                                                        # Asignamos un umbral para filtrar objetos parecidos
    ubi = np.where(deteccion >= umbral)                                  # La ubicacion de los objetos la vamos a guardar cuando supere el umbral
    for pt in zip (*ubi[::-1]):                                          # Creamos un for para dibujar todos los rectangulos
        cv.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255,0,0), 1)         # Dibujamos los n rectangulos que hayamos identificado con el tamaño del recorte y de color


#simbolos de interes
path = 'D:\\DocumentosD\\Python\\Vision-artificial\\Imagenes\\Train' #Ruta donde se encuentran las imagenes de referencia
ext = '.jpg'
r1 = cv.imread(Ruta + 'r_pills' + ext)
cv.imshow('r1',r1)
cap=cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == True:
        detect(frame,r1)
        cv.imshow('frame', frame)
        
        if cv.waitKey(1) & 0xFF == ord('s'):
            break
        
cap.release()
cv.destroyAllWindows()
