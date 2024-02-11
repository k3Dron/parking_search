import cv2
import matplotlib.pyplot as plt
import numpy as np
#from functions import get_roi, check_for_vehicles

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)
classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').rsplit('\n')  
model.setInputSize(360,360)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

object_to_be_detected = [2, 3, 4, 5, 6, 7, 8]
objects_observable = np.array(object_to_be_detected)

def get_roi(connected_components):
    print("mid test")
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totalLabels) :
        x1 = int(values[i, cv2.CC_STAT_LEFT]*coef)
        y1 = int(values[i, cv2.CC_STAT_TOP]*coef)
        w = int(values[i, cv2.CC_STAT_WIDTH]*coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT]*coef)

        slots.append([x1, y1, w, h])
    return slots

def check_for_vehicles(spot_bgr, frame):
    ClassIndex, confidence, bbox = model.detect(spot_bgr, confThreshold = 0.55)
    if(len(ClassIndex) != 0) :
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if(ClassInd<=80) :
                    cv2.rectangle(spot_bgr,boxes,(255, 0, 0), 2)
                    cv2.putText(spot_bgr,classLabels[ClassInd - 1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color= (0, 255, 0))
    cv2.imshow('Object Detection', frame)     
    for val in objects_observable:
        if val in ClassIndex:
            return True
    return False   

mask = './mask_1920_1080.png'
mask1 = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(mask1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
connected_components = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
slots = get_roi(connected_components)
cap = cv2.VideoCapture(0)
ret = True
while ret:
    ret, frame = cap.read()
    for slot in slots:
        x1,y1,w,h = slot
        slot_crop = frame[y1:y1+h, x1:x1+w, :]
        slot_status = check_for_vehicles(slot_crop, frame)
        if slot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
    cv2.imshow('Object Detection', frame) 
    if(cv2.waitKey(2) & 0xFF) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()               