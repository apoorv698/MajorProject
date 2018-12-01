import cv2
import time
import os
import numpy as np

from keras import backend as k
from keras.models import load_model

img_w = 75
img_h = 75
weight_dir = os.path.join(os.getcwd(), 'weights/')
model_name = 'face_model.h5'
model_dir = os.path.join(os.getcwd(), 'models/')

def predict(model, pic):
    global weight_dir
    gray_img = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (img_w,img_h))
    gray_img = np.expand_dims(gray_img, axis=2)
    gray_img = np.array([gray_img])
    sum=0.0
    counter=0.0
    print("Pridicting...")
    for wt in os.listdir(weight_dir):
        counter+=1.0
        model.load_weights(weight_dir+wt)
        ynew = model.predict_classes(gray_img)
        sum+=ynew[0]
    predicted_age = sum/counter
    return predicted_age


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    cap = cv2.VideoCapture(0) 
    model = load_model(model_dir+model_name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,400)
    fontScale = 1
    fontColor = (255,255,255)
    lineType = 2
    while 1:  
      
        # reads frames from a camera 
        ret, img = cap.read()  
      
        # convert to gray scale of each frames 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
      
        # Detects faces of different sizes in the input image 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        roi = None
        for (x,y,w,h) in faces: 
            # To draw a rectangle in a face  
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
            roi = img[y:y+h, x:x+w]      
        # Display an image in a window 
        try:
            age = predict(model, roi)
            print("Approximate face Age is "+str(age))
            cv2.putText(img,'Approx. Age = '+str(age), bottomLeftCornerOfText,font, fontScale, fontColor, lineType)
            cv2.imshow('Age',img)
        except Exception as e:
            cv2.imshow('Age', img)
        # Wait for Esc key to stop 
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
      
    # Close the window 
    cap.release() 
      
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows()  