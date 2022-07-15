#D:\DocumentosD\UTP\brazo\prueba.jpg

import numpy as np
import cv2  
import matplotlib.pyplot as plt
import os

#detect images 
path = 'D:\\DocumentosD\\Python\\Vision-artificial\\Imagenes\\TrainF'
images = []
classNames = [] 

colors = [[],[],[],[]] #[gray, b, g, r]
not_colors = [[],[],[],[]]
color_name = ['Gray', 'Blue', 'Green', 'Red']
myList = os.listdir(path) 

print('Total classes detected: ', len(myList)) #Imprime la cantidad de archivos que se encuentran en la ruta


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

def histogram(img, plus_name, img_name, hist_name, row, col):
    for i in range(len(img[0])):
        fig = plt.figure('Histograma ' + plus_name + img_name[i])
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        
        for j in range(len(img)):
            hist = cv2.calcHist([img[j][i]], [0], None, [256], [0,256])
            ax = fig.add_subplot(row, col, j+1)
            ax.plot(hist)
            ax.set_xlabel("Bins")
            ax.set_ylabel("# de pixeles")
            ax.set_title("Histograma " + hist_name[j])
            ax.grid(color='gray', linestyle='dashed', linewidth=1, alpha=0.4)
            # Pintar los ejes pasando por (0,0)
            ax.axhline(0, color='black', linewidth=0.5)
            
def bin(img, u):
    bw = [[],[],[],[]]
    for i in range(len(img[0])):  
        for j in range(len(img)):
            threshold, thresh = cv2.threshold(img[j][i], u, 255,cv2.THRESH_BINARY)
            bw[j].append(thresh)
    return bw

def blur_img(img, u):
    br = [[],[],[],[]]
    for i in range(len(img[0])):  
        for j in range(len(img)):
            #b = cv2.GaussianBlur(img[j][i], (u,u), cv2.BORDER_DEFAULT)
            b = cv2.medianBlur(img[j][i], u)
            br[j].append(b)
    return br

def morph(img, opcion, size_kernel):
    mp = [[],[],[],[]]
    kernel = np.ones((size_kernel, size_kernel), np.uint8)
    for i in range(len(img[0])):  
        
        for j in range(len(img)):
            if opcion == 0:  
                a = cv2.morphologyEx(img[j][i], cv2.MORPH_OPEN, kernel)
            elif opcion == 1:
                a = cv2.morphologyEx(img[j][i], cv2.MORPH_CLOSE, kernel)
            elif opcion == 2:
                a = cv2.erode(img[j][i], kernel, iterations=1)
            elif opcion == 3:
                a = cv2.dilate(img[j][i], kernel, iterations=1)
            else:
                a = 0    
            mp[j].append(a)
       
    return mp


def show_img(img, name, plus_name):
    for i in range(len(img[0])):
        h1, h2 = [], []
        nh1, nh2 = [], []
        for j in range(len(img)):
            if j <= 1:
                h1.append(img[j][i])
            else:
                h2.append(img[j][i])
            
        show = cv2.resize( np.vstack( (rz(np.hstack( h1 ), width=1300), rz(np.hstack( h2 ), width=1300)) ), (800,600), interpolation=cv2.INTER_AREA )
        cv2.imshow(name[i] + plus_name, show)
        


for cl in myList: 
    imgCur = rz(cv2.imread(f'{path}\\{cl}'), height=800) 
    b, g, r = cv2.split(imgCur)
    gy = cv2.cvtColor(imgCur, cv2.COLOR_BGR2GRAY)
    images.append(imgCur) 
    colors[0].append(gy)
    colors[1].append(b)
    colors[2].append(g)
    colors[3].append(r)
    not_colors[0].append(cv2.bitwise_not(gy))
    not_colors[1].append(cv2.bitwise_not(b))
    not_colors[2].append(cv2.bitwise_not(g))
    not_colors[3].append(cv2.bitwise_not(r))
    classNames.append(os.path.splitext(cl)[0]) 
print(classNames)

print('colors[0]: ', len(colors[0]), ' | colors: ', len(colors))


#colores

#show_img(not_colors, classNames, 'inverso')   
#histogram(not_colors,'inverso ', classNames, color_name, row=2, col=2)
bw = bin(not_colors, u = 140) 
mp = morph(bw, opcion=1, size_kernel=20) 
mp = morph(mp, opcion=3, size_kernel=10) 
blur = blur_img(mp, u=9)
#show_img(bw, classNames, ' bin')   
show_img(mp, classNames, ' morph2')  
show_img(blur, classNames, ' blur')

    
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()


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





# # cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# # im = cv2.imread(path)                    # Read image
# # imS = cv2.resize(im, (960, 540))                # Resize image

# # cv2.imshow("output", imS)                       # Show image
# # cv2.waitKey(0)   


# fig1 = plt.figure('Histogramas')
# fig1.subplots_adjust(hspace=0.5, wspace=0.5)

# for i in range(len(gray)):
    
#     hist = cv2.calcHist([gray[i]], [0], None, [256], [0,256])
#     ax = fig1.add_subplot(1, 1, i+1)
#     ax.plot(hist)
#     ax.set_xlabel("Bins")
#     ax.set_ylabel("# de pixeles")
#     ax.set_title("Histograma imagen " + str(i))
#     ax.grid(color='gray', linestyle='dashed', linewidth=1, alpha=0.4)
#     # Pintar los ejes pasando por (0,0)
#     ax.axhline(0, color='black', linewidth=0.5)
    
# plt.show()
# cv2.waitKey()
# cv2.destroyAllWindows()














