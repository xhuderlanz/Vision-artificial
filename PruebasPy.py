import cv2

#open cv DNN

net=cv2.dnn.readNet("C:/Users/Slash/Desktop/UTP/VII Semestre/Sensores y actuadores/dnn_model/yolov4-tiny.weights", "C:/Users/Slash/Desktop/UTP/VII Semestre/Sensores y actuadores/dnn_model/yolov4-tiny.cfg")
model=cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

#Cargar la lista de objetos
classes=[]
with open("C:/Users/Slash/Desktop/UTP/VII Semestre/Sensores y actuadores/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name=class_name.strip()
        classes.append(class_name)

print("Lista de objetos")
print(classes)

#Iniciar camara
cap=cv2.VideoCapture(0)

while True:
 #Conseguir los frames   
 ret, frame=cap.read()

 #Deteccion de objetos
 (class_ids, scores, bboxes)=model.detect(frame)
 for class_id, score, bbox in zip(class_ids, scores, bboxes):
    (x, y, w, h)=bbox
    class_name=classes[class_id]

    cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
    cv2.putText(frame, "Precision:", (x+110, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
    cv2.putText(frame, str(score), (x+200, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 0, 50), 3)
 


 cv2.imshow("Frame", frame)
 cv2.waitKey(1)