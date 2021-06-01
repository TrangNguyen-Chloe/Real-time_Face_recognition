import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

#load detection model
MODEL_d = 'D:\Facedetection\yolo\yolov3-face.cfg'
WEIGHT = 'D:\Facedetection\yolo\yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL_d, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#prediction model
MODEL_p = load_model('FaceRecognition_model1.h5')

#data folder path
data_dir = 'D:\Facedetection\data'
name_list = os.listdir(data_dir)
name_list.sort()
labels = {i:name_list[i] for i in range(len(name_list))}
# index_to_label = {0: 'Albert', 1: 'Chinh', 2: 'Chloe', 3: 'Ly', 4: 'Viet'}


cap = cv2.VideoCapture(0)
#webcam check
if not cap.isOpened():
    raise IOError('Cannot open webcam')

#capture frame by frame
while True:
    ret, frame = cap.read()
#detect face with yolo
    #get detection
    
    IMG_WIDTH, IMG_HEIGHT = 416, 416
    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    #Choosing high confidence detection
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
    # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with #high confidence)
            if confidence > 0.5:
                print(confidence)
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                
    # Find the top left point of the bounding box 
                topleft_x = int(center_x - width/2)
                topleft_y = int(center_y - height/2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #Display bounding box and text
    result = frame.copy()
    final_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)
        # Extract position data
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        # Draw bouding box with the above measurements
        result = cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 1)
        face_frame = result[top:top+height, left:left+width] 
#make prediction with trained model
        face_frame = cv2.resize(face_frame, (150,150))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis = 0) 
        face_frame = preprocess_input(face_frame)
        prediction = MODEL_p.predict(face_frame, batch_size = 10)
        label = np.argmax(prediction, axis = 1)
        name = labels[int(label)]
        prop = round(max(prediction[0])*100)

        text = f'{name}, {prop}%'
#display prediction & probability
        cv2.putText(result, text, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=2)

# Display text about number of detected faces on topleft corner
    text2 = f'Number of faces detected: {len(indices)}'
    coor2 = (20, 25)
    cv2.putText(result, text2, coor2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

#showing final result
    cv2.imshow('Input', result)

    c = cv2.waitKey(1) #wait for 1ms
    #Break when pressing ESC and capture when press space
    if c == 27:
        break
    elif c == 32:
        img_name = f'F{j}.jpg'
        crop = result[top:top+height, left:left+width] 
        img_path = os.path.join('D:\Facedetection\data\Chloe',img_name)
        cv2.imwrite(img_path, crop)

cap.release()
cv2.destroyAllWindows()
    
    







