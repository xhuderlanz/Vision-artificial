
import cv2 as cv
import numpy as np

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



#Leer imagen
img = rz(cv.imread('D:\\DocumentosD\\UTP\\brazo\\Train2\\Toilet_paper.jpeg',cv.IMREAD_COLOR), width= 400 )

#Mantener original y trabajar con copia
show_img = np.copy(img)
img_size = img.shape[0], img.shape[1]

print('tamaño: ', img_size)

selected_pts = []

#Mouse callback
def mouse_callback(event, x, y, flags, param):
    global selected_pts, show_img
    
    if event == cv.EVENT_LBUTTONUP:
        #anexar al arreglo el punto actual
        selected_pts.append([x, y])
        #dibujar un pequeño circulo donde se clico
        cv.circle(show_img, (x, y), 10, (0, 255, 0), 3)
        

#Definición de la función select_points
def select_points(image, points_num):
    global selected_pts
    selected_pts = []
    
    cv.namedWindow('image')
    cv.setMouseCallback('image', mouse_callback)
    
    while True:
        cv.imshow('image', image)
        
        k = cv.waitKey(1)
        
        if k == 27 or len(selected_pts) == points_num:
            break
    cv.destroyAllWindows()
    
    return np.array(selected_pts, dtype=np.float32) 


# Affine transformation
#
# getAffineTransform takes two arguments
# the source points and the destination points for those selected pts


#get the selected points
src_pts = select_points(show_img, 3)
print(src_pts)

#Destination points
dts_pts = np.array([ [0, img_size[0]], [0, 0], [img_size[0], 0]], dtype=np.float32)
print(dts_pts)


#Apply affine transform
affine_m = cv.getAffineTransform(src_pts, dts_pts)
unwarped_img = cv.warpAffine(img, affine_m, (img_size[0], img_size[0]))

cv.imshow('result', np.hstack((show_img, unwarped_img)))
cv.waitKey()
cv.destroyAllWindows()


# Perspective transformation

#get the selected points
show_img = np.copy(img) 
src_pts = select_points(show_img, 4)
print(src_pts)

#Destination points
dts_pts = np.array([ [0, img_size[0]], [0, 0], [img_size[0], 0], [img_size[0], img_size[0]]], dtype=np.float32)
print(dts_pts)

perspective_m = cv.getPerspectiveTransform(src_pts, dts_pts)
unwarped_img = cv.warpPerspective(img, perspective_m, (img_size[0], img_size[0]))

cv.imshow('result', np.hstack((show_img, unwarped_img)))
cv.waitKey()
cv.destroyAllWindows()