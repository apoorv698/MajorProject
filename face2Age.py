import cv2
import time
import os

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.metrics import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array

img_channels = 1
img_w = 75
img_h = 75
classes = 93
filters = 32
pool = 3
conv = 3
chanDim = -1
weight_dir = 'weights/'

def predict(model, pic):
    global weight_dir
    gray_img = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (img_w,img_h))
    gray_img = np.expand_dims(gray_img, axis=2)
    gray_img = np.array([gray_img])
    sum=0.0
    counter=0.0
    for wt in os.listdir(weight_dir):
        counter+=1.0
        model.load_weights(weight_dir+wt)
        ynew = model.predict_classes(gray_img)
        sum+=ynew[0]
    predicted_age = sum/counter
    return predicted_age


def make_model():
    global filters,conv, img_w, img_h, img_channels, pool
    model = Sequential()
    model.add(Conv2D(filters, (conv, conv), padding="same", input_shape=(img_w, img_h, img_channels)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    model.add(Dropout(0.2))

    pool -= 1
    filters *= 2
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    model.add(Dropout(0.2))

    filters *= 2
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    model.add(Dropout(0.2))

    filters *= 2
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    model.add(Dropout(0.2))

    filters *= 4
    model.add(Flatten())
    model.add(Dense(filters))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # print(model.summary())

    return model

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    cap = cv2.VideoCapture(0) 
    model = make_model()
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
        except:
            cv2.imshow('Age', img)
        # Wait for Esc key to stop 
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
      
    # Close the window 
    cap.release() 
      
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows()  