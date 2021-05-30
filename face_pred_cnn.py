from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX

#from keras.preprocessing import image
model = load_model('Avijit_cnn.h5')
ResultMap=['Avijit', 'Bakul']

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Doing some Face Recognition with the webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    _, frame = cam.read()
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = frame[y:y+h, x:x+w]
        face = cv2.resize(cropped_face, (64, 64))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        id=np.argmax(pred)
        name = ResultMap[id]
        
    
    
   
         
         
        cv2.putText(frame,name, (x+5,y-5), font, 1, (255,255,255),)
        #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1 )  
    '''if type(face) is np.ndarray:
        face = cv2.resize(face, (64, 64))
        im = Image.fromarray(face, 'RGB')
           #Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)
        result=model.predict(img_array,verbose=0)
        print('Prediction is: ',ResultMap[np.argmax(result)])
        name=ResultMap[np.argmax(result)]
        cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
       
                     
        name="None matching"
        
        if(pred[0][1]>0.5):
            name='Bakul'
            cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        elif(pred[0][0]>0.5): 
            name='Avijit'
            cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)'''
        
    #else:
        #cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()