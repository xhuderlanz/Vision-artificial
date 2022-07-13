

import cv2
import numpy as np

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


#1. Leer la imagen de referencia de objeto de interés
ref_img = cv2.imread("D:\\DocumentosD\\UTP\\brazo\\pinguiperre\\Minion (1).jpeg", cv2.IMREAD_COLOR)
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

#2. Leer la imagen de objeto en una escena desordenada
scene_img = cv2.imread("D:\\DocumentosD\\UTP\\brazo\\pinguiperre\\Minion (2).jpeg", cv2.IMREAD_COLOR)
scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

#3. Detectar feature points
ref_points = cv2.SIFT_create(1000)
ref_keypoints, ref_descriptors = ref_points.detectAndCompute(ref_img, None)
print("# Keypoints REF: {}, Descriptor REF: {}".format(len(ref_keypoints), ref_descriptors.shape))

scene_points = cv2.SIFT_create(1000)
scene_keypoints, scene_descriptors = scene_points.detectAndCompute(scene_img, None)
print("# Keypoints SCENE: {}, Descriptor SCENE: {}".format(len(scene_keypoints), scene_descriptors.shape))

#4. Extracción de descriptores

#5. Hacer el match usando los descriptores
bf = cv2.BFMatcher()
matches = bf.knnMatch(ref_descriptors, scene_descriptors, k=2)

#Se filtran los puntos utilizando la ratio test
good_points = []
for m, n in matches:
    if m.distance < 0.4*n.distance:
        good_points.append(m)

#6. RECUADRO
#Pintar los puntos encontrados en las imágenes
matched_img = cv2.drawMatches(ref_img, ref_keypoints, scene_img, scene_keypoints, good_points, scene_img, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matched features", rz(matched_img,width=800))
cv2.waitKey()
cv2.destroyAllWindows()