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
weight_dir = os.path.join(os.getcwd(), 'weights/') 
model_name = 'face_model.h5'
model_dir = os.path.join(os.getcwd(), 'models/')
testing_dir = os.path.join(os.getcwd(), 'testing-image/')

def prepare_data():
    train_path = 'C:/Users/apoov/face2AgeDetect-Major/UTKFace/'
    training_img = []
    training_label =[]

    for num, filename in enumerate(os.listdir(train_path)):
        if int(filename.split('_')[0])>92:
            continue
        img = cv2.imread(train_path+filename)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (img_w,img_h))
        gray_img = np.expand_dims(gray_img, axis=2)
        training_img.append(img_to_array(gray_img))
        age = int(filename.split('_')[0])
        # age /=7
        # age =round(age)*7
        # training_label.append(age/7)
        training_label.append(age)
        if num % 1000 == 0:
            print('{} files completed.'.format(num))
    print('--Extraction Complete.--')
    X = np.array(training_img, dtype=float)/255.0
    y = np.array(training_label)
    np.resize(X,(1,img_w,img_h,1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    return X_train, X_test, y_train, y_test

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

def make_pridiction(model):
    count=0
    t_error=0
    for filename in os.listdir(testing_dir):
        if filename.split('_')[0]=='1':
            continue
        pic = cv2.imread(testing_dir+filename)
        gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        roi = None
        for (x,y,w,h) in faces: 
            roi = gray[y:y+h, x:x+w]
        try:
            print('using face only')
            gray_img = cv2.resize(roi, (img_w,img_h))
            gray_img = np.expand_dims(gray_img, axis=2)
            gray_img = np.array([gray_img])/255.0
        except:
            print('Unable to find face')
            print('using whole pic')
            gray = cv2.resize(gray, (img_w,img_h))
            gray = np.expand_dims(gray, axis=2)
            gray = np.array([gray])/255.0
        # y_chk = np.array([round(21/7)])
        # Y_chk = np_utils.to_categorical(y_chk, classes)

        # print(gray_img.shape, Y_chk.shape)
        sum=0.0
        counter=0.0
        for wt in os.listdir(weight_dir):
            counter+=1.0
            model.load_weights(weight_dir+wt)
            # score = model.evaluate(gray_img, Y_chk, verbose=0)
            ynew=0
            try:
                ynew = model.predict_classes(gray_img)
            except:
                ynew = model.predict_classes(gray)
            sum+=ynew[0]
        # show the inputs and predicted outputs
        predicted_age = sum/counter
        
        try:
            error = round(abs(int(filename.split('_')[0]) - predicted_age),2)
            print("Predicted Age = %s with error = %s of yrs for image %s" % (round(predicted_age), error,filename))
            t_error += abs(int(filename.split('_')[0]) - predicted_age)
            count+=1
            # if error>3:
            #     os.remove(testing_dir+filename)
            #     print('deleted!!')
        except:
            print("Predicted Age = %s for image %s" % (round(predicted_age), filename))

        print('Total error: ',t_error,'Avg error: ',t_error/count)



if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

    if len(os.listdir(weight_dir))==0:

        data_dir = os.path.join(os.getcwd(), 'data')

        X_train_path = os.path.join(data_dir, 'X_train.npy')
        X_test_path = os.path.join(data_dir, 'X_test.npy')
        y_train_path = os.path.join(data_dir, 'y_train.npy')
        y_test_path = os.path.join(data_dir, 'y_test.npy')

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

        if len(os.listdir(model_dir))==0:
            model = make_model()
            model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
            model.save(model_dir+model_name)
            print('--Model prepared.--')
        else:
            print('--Using saved Model.--')
            model = load_model(model_dir+model_name)

        # filepath="weights_best.hdf5"
        filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        
        if len(os.listdir(weight_dir))==0:
            checkpoint = ModelCheckpoint(weight_dir+filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, callbacks=callbacks_list, verbose=1, validation_data=(X_test, Y_test))
            plot_graphs(history)
    else:

        if len(os.listdir(model_dir))==0:
            model = make_model()
            model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
            model.save(model_dir+model_name)
            print('--Model prepared.--')
        else:
            print('--Using saved Model.--')
            model = load_model(model_dir+model_name)


    print('--Making Predictions.--')

    make_pridiction(model)