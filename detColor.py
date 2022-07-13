import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt

def dibujar(mask, color):
    contornos, hierc = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for c in contornos:
        area = cv.contourArea( c )
        if area > 1000:
            M=cv.moments(c )
            if (M["m00"] == 0): M["m00"] = 1
            x = int(M["m10"]/M["m00"])
            y = int(M["m01"]/M["m00"])
            cv.circle(frame, (x, y), 7, (0, 255), -1)
            font = cv.FONT_HERSHEY_SIMPLEX
            if color == (0, 255, 0):
                titleColor = 'Verde'
            if color == (255, 0, 0):
                titleColor = 'Azul'
            if color == (0, 0, 255):
                titleColor = 'Rojo'
            if color == (0, 255, 255):
                titleColor = 'Amarillo'
            cv.putText(frame, '{},{}'.format(x, y), (x+10, y), font, 0.75, (0, 255, 0), 1, cv.LINE_AA)
            cv.putText(frame, titleColor  , (x+10, y-30), font, 0.60, (0, 255, 0), 2, cv.LINE_AA)

            nuevoContorno = cv.convexHull( c)
            cv.drawContours(frame, [nuevoContorno] , 0, color, 3)
    

cap=cv.VideoCapture(0)

lowSvalue = 190
highSvalue = 255

lowVvalue = 20
highVvalue = 255

blueBajo = np.array([100, lowSvalue, lowVvalue], np.uint8)
blueAlto = np.array([125, highSvalue, highVvalue], np.uint8)

greenBajo = np.array([46, lowSvalue-100, lowVvalue], np.uint8)
greenAlto = np.array([85, highSvalue, highVvalue], np.uint8)

yellowBajo = np.array([15, lowSvalue, lowVvalue], np.uint8)
yellowAlto = np.array([45, highSvalue, highVvalue], np.uint8)


redBajo1 = np.array([0, lowSvalue, lowVvalue], np.uint8)
redAlto1 = np.array([8, highSvalue, highVvalue], np.uint8)

redBajo2 = np.array([175, lowSvalue, lowVvalue], np.uint8)
redAlto2 = np.array([179, highSvalue, highVvalue], np.uint8)

font = cv.FONT_HERSHEY_SIMPLEX

    
while True:
    ret, frame = cap.read()
    if ret == True:
        frameHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        maskBlue = cv.inRange(frameHSV, blueBajo, blueAlto)
        maskYellow = cv.inRange(frameHSV, yellowBajo, yellowAlto)
        maskRed1 = cv.inRange(frameHSV, redBajo1, redAlto1)
        maskRed2 = cv.inRange(frameHSV, redBajo2, redAlto2)
        maskRed = cv.add(maskRed1, maskRed2)
        maskGreen = cv.inRange(frameHSV, greenBajo, greenAlto)
        
        dibujar(maskBlue,(255,0,0))
        dibujar(maskYellow,(0,255,255))
        dibujar(maskRed,(0,0,255))
        dibujar(maskGreen,(0,255,0))
        cv.imshow('frame', frame)
        
        if cv.waitKey(1) & 0xFF == ord('s'):
            break
        
cap.release()
cv.destroyAllWindows()