#D:\DocumentosD\UTP\brazo\prueba.jpg

import numpy as np
import cv2  
import matplotlib.pyplot as plt
import os


def rz(image, width=None, height=None, inter=cv2.INTER_AREA):
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

#detect images 
path = 'D:\\DocumentosD\\Python\\Vision-artificial\\Imagenes\\TrainF'
images, gray, B_images, G_images, R_images= [], [], [], [], [] 
classNames = [] 
myList = os.listdir(path) 

print('Total classes detected: ',len(myList)) #Imprime la cantidad de archivos que se encuentran en la ruta

for cl in myList: 
    imgCur = rz(cv2.imread(f'{path}\\{cl}'), height=800) 
    b, g, r = cv2.split(imgCur)
    images.append(imgCur) 
    gray.append(cv2.cvtColor(imgCur, cv2.COLOR_BGR2GRAY))
    B_images.append(b)
    G_images.append(g)
    R_images.append(r)
    classNames.append(os.path.splitext(cl)[0]) 
print(classNames)




# sift = cv2.SIFT_create(nfeatures=500)

# kp, des, kp_images  = [], [], []

# for i in range(len(images)):
#     k, d = sift.detectAndCompute(images[i], None)
#     kp_images.append(cv2.drawKeypoints(images[i], k, None))
#     kp.append(k)
#     des.append(d)

# himages, hkp_images = [], []
# for i in range(len(images)):
#     himages.append(images[i])
#     hkp_images.append(kp_images[i])

# show = cv2.resize( np.vstack( (rz(np.hstack( himages ), width=1300), rz(np.hstack( hkp_images ), width=1300)) ), (1280,720), interpolation=cv2.INTER_AREA )

# cv2.imshow('Imagen', show)


fig1 = plt.figure('Histogramas')
fig1.subplots_adjust(hspace=0.5, wspace=0.5)

for i in range(len(gray)):
    
    hist = cv2.calcHist([gray[i]], [0], None, [256], [0,256])
    ax = fig1.add_subplot(1, 1, i+1)
    ax.plot(hist)
    ax.set_xlabel("Bins")
    ax.set_ylabel("# de pixeles")
    ax.set_title("Histograma imagen " + str(i))
    ax.grid(color='gray', linestyle='dashed', linewidth=1, alpha=0.4)
    # Pintar los ejes pasando por (0,0)
    ax.axhline(0, color='black', linewidth=0.5)
    
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()


# # cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# # im = cv2.imread(path)                    # Read image
# # imS = cv2.resize(im, (960, 540))                # Resize image
# # cv2.imshow("output", imS)                       # Show image
# # cv2.waitKey(0)   


