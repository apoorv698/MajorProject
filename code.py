# importing required Modules

import cv2  
import os

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras import backend as k
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.metrics import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array

# Constants Declaration

img_channels = 1
img_w = 75
img_h = 75
batch_size = 32
classes = 93
epoch = 20
filters = 32
pool = 3
conv = 3
chanDim = -1
model_name = 'face_model.h5'

# storing different path
weight_dir = os.path.join(os.getcwd(), 'weights/')
model_dir = os.path.join(os.getcwd(), 'models/')
testing_dir = os.path.join(os.getcwd(), 'testing-image/')
train_path = 'C:/Users/apoov/face2AgeDetect-Major/UTKFace/'
data_dir = os.path.join(os.getcwd(), 'data')

# Loading training dataset, processing images and converting into numpy arrays for testing and training of the model
def prepare_data():
    global train_path
    training_img = []
    training_label = []

    for num, filename in enumerate(os.listdir(train_path)):
        # Getting each file from training dataset
        if int(filename.split('_')[0])>92:
            continue
        img = cv2.imread(train_path+filename)

        # Processing the 'img' image file
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img = cv2.equalizeHist(gray_img)
        gray_img = cv2.resize(gray_img, (img_w,img_h))
        gray_img = np.expand_dims(gray_img, axis=2)

        # populating the training image set 'training_img' with labels set in 'training_label'
        training_img.append(img_to_array(gray_img))
        age = int(filename.split('_')[0])
        training_label.append(age)
        if num % 1000 == 0:
            print('{} files completed.'.format(num))
    print('--Extraction Complete.--')

    # Normalization of training image set
    X = np.array(training_img, dtype=float)/255.0
    y = np.array(training_label)

    # resizing according to need
    np.resize(X,(1,img_w,img_h,1))

    # populating test and train dataset with label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    return X_train, X_test, y_train, y_test

# 'make_model()' makes the Convolutional Neural Network 
def make_model():
    global filters,conv, img_w, img_h, img_channels, pool
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters, (conv, conv), padding="same", input_shape=(img_w, img_h, img_channels)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.2))

    pool -= 1
    filters *= 2
    # 2nd Convolutional Layer
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.2))

    filters *= 2
    # 4th Convolutional Layer
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.2))

    filters *= 2

    # 6th Convolutional Layer
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    # 7th Convolutional Layer
    model.add(Conv2D(filters, (conv, conv), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.2))

    filters *= 4
    # Passing it to a Fully Connected layer
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(filters))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

# Only displayed when model is trained
def plot_graphs(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# function to select image from test samples and finding the predicted age and finding the total error after the process 
def make_pridiction(model):
    count=0
    t_error=0

    
    for filename in os.listdir(testing_dir):
        # Getting each file from training dataset
        pic = cv2.imread(testing_dir+filename)

        # Processing the 'img' image file
        gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Loading 'haarcascade_frontalface_default.xml'  to find faces 
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # finding region of image that contain face
        roi = None
        for (x,y,w,h) in faces: 
            roi = gray[y:y+h, x:x+w]
        
        # checking if roi contains some face or not
        # ie. if some face have been detected in the image

        try:
            # print('using face only')
            gray_img = cv2.resize(roi, (img_w,img_h))
            gray_img = np.expand_dims(gray_img, axis=2)
            gray_img = np.array([gray_img])/255.0
        except:
            # print('Unable to find face')
            # print('using whole pic')
            gray = cv2.resize(gray, (img_w,img_h))
            gray = np.expand_dims(gray, axis=2)
            gray = np.array([gray])/255.0

        sum=0.0
        counter=0.0
        
        # Loading the weights and predicting the class ie the age of the preson
        for wt in os.listdir(weight_dir):
            counter+=1.0
            model.load_weights(weight_dir+wt)
            ynew=0
            try:
                ynew = model.predict_classes(gray_img)
            except:
                ynew = model.predict_classes(gray)
            sum+=ynew[0]
        
        # finding predicted age
        predicted_age = sum/counter
        
        # show the inputs and predicted outputs with absolute error
        try:
            error = round(abs(int(filename.split('_')[0]) - predicted_age),2)
            print("Predicted Age = %s with error = %s of yrs for image %s" % (round(predicted_age), error,filename))
            t_error += abs(int(filename.split('_')[0]) - predicted_age)
            count+=1
        except:
            print("Predicted Age = %s for image %s" % (round(predicted_age), filename))

    print('Avg error: ',t_error/count)



if __name__ == '__main__':

    # will only display the errors for tensorflow and no othe info message will be displayed
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

    # creating new weight dir if it doesn't exists
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    # if weight_dir is empty ie model is not trained
    # thus whole process must begin from the start
    if len(os.listdir(weight_dir))==0:

        # getting the dataset paths
        X_train_path = os.path.join(data_dir, 'X_train.npy')
        X_test_path = os.path.join(data_dir, 'X_test.npy')
        y_train_path = os.path.join(data_dir, 'y_train.npy')
        y_test_path = os.path.join(data_dir, 'y_test.npy')

        # if path doesnot exist then preprocessing must done by calling prepare_data() 
        # otherwise data must be loaded in testing and training samples
        if not os.path.exists(data_dir):
            print('--Processed data does not exists. Preparing data.--')
            os.mkdir(data_dir)
            X_train, X_test, y_train, y_test = prepare_data()
            print('--Preprocessing Complete.--')
            np.save(X_train_path, X_train)
            np.save(X_test_path, X_test)
            np.save(y_train_path, y_train)
            np.save(y_test_path, y_test)
        else:
            print('--Loading Processed data.--')
            X_train = np.load(X_train_path)
            X_test = np.load(X_test_path)
            y_train = np.load(y_train_path)
            y_test = np.load(y_test_path)

        Y_train = np_utils.to_categorical(y_train, classes)
        Y_test = np_utils.to_categorical(y_test, classes)

        print('--Datasets prepared.--')

        # preparing the model, if it does not exists then call make_model() and save the model in model_dir
        # else just load the saved model

        if len(os.listdir(model_dir))==0:
            model = make_model()
            model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
            model.save(model_dir+model_name)
            print('--Model prepared.--')
        else:
            print('--Using saved Model.--')
            model = load_model(model_dir+model_name)

        filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

        # finally training the model using the processed data
        checkpoint = ModelCheckpoint(weight_dir+filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, callbacks=callbacks_list, verbose=1, validation_data=(X_test, Y_test))

        # calling the plot_graph() functions
        plot_graphs(history)
    # else if trained weights exist thus model also exist
    # thus model is loaded
    else:
        print('--Using saved Model.--')
        model = load_model(model_dir+model_name)

    # finally making the predictions using the model
    print('--Making Predictions.--')

    make_pridiction(model)