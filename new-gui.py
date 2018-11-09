import queue 
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as ttk
import time
import threading
from tkinter.ttk import Progressbar
import cv2
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


class GUI:
	def __init__(self,root):
		self.filename = None
		self.root = root
		self.panel = None
		self.root.geometry('620x400+300+300')
		self.root.resizable(0, 0)

		self.label_info = Label(self.root, text='Status: \nWelcome!!')
		self.label_info.grid(column=1,row=3, padx=5, pady=3, sticky=S)

		self.btn_select_image = Button(self.root, text="Select Image", command=self.selectImage)
		self.btn_select_image.grid(column=4,row=3, padx=3, pady=3, sticky=S)

		self.btn_predict = Button(self.root, text="Predict Age", command=self.age_prediction)
		self.btn_predict.grid(column=5,row=3, padx=3, pady=3, sticky=S)

		self.btn_quit = Button(self.root, text='Quit', command = lambda : self.root.destroy())
		self.btn_quit.grid(column=6,row=3, padx=3, pady=3, sticky=S)

		self.filename = 'upload-image/sample-image.jpg'
		image = Image.open(self.filename)
		baseheight = 200
		hpercent = (baseheight / float(image.size[1]))
		wsize = int((float(image.size[0]) * float(hpercent)))
		image = image.resize((wsize, baseheight), Image.ANTIALIAS)
		image = ImageTk.PhotoImage(image)
		self.panel = Label(image=image)
		self.panel.image = image
		self.panel.grid(column=3, row=1, columnspan=2)

		self.label_predicted_age = Label(self.root, text='Predicited Age-')
		self.label_predicted_age.grid(column=1,row=2, padx=5, pady=3, sticky=S)
		self.label_predicted_age_val = Label(self.root, text=str(None))
		self.label_predicted_age_val.grid(column=2,row=2, padx=5, pady=3, sticky=S)
		self.label_actual_age = Label(self.root, text='Actual Age-')
		self.label_actual_age.grid(column=3,row=2, padx=5, pady=3, sticky=S)
		self.label_actual_age_val = Label(self.root, text=str(None))
		self.label_actual_age_val.grid(column=4,row=2, padx=5, pady=3, sticky=S)
		self.label_error = Label(self.root, text='Error-')
		self.label_error.grid(column=5,row=2, padx=5, pady=3, sticky=S)
		self.label_error_val = Label(self.root, text=str(None))
		self.label_error_val.grid(column=6,row=2, padx=5, pady=3, sticky=S)

		self.root.grid_columnconfigure(3, minsize=100)
		self.root.grid_rowconfigure(3, minsize=80)
		self.root.grid_columnconfigure(2, minsize=100)
		self.root.grid_rowconfigure(2, minsize=80)
		self.root.grid_columnconfigure(1, minsize=100)
		self.root.grid_rowconfigure(1, minsize=140)

		self.root.wm_title("Age Predictor")
		self.model = None
		self.make_model()
		self.root.mainloop()

	def predict_with_preprocessing(self, img):
		global weight_dir, img_w, img_h
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
			print(gray_img.shape)
		except:
			print('Unable to find face')
			print('using whole picture')
			gray = cv2.resize(gray, (img_w,img_h))
			gray = np.expand_dims(gray, axis=2)
			gray = np.array([gray])/255.0
			print(gray.shape)

		sum=0.0
		counter=0.0
		for wt in os.listdir(weight_dir):
			counter+=1.0
			self.model.load_weights(weight_dir+wt)
			try:
				ynew = self.model.predict_classes(gray_img)
			except:
				ynew = self.model.predict_classes(gray)
			sum+=ynew[0]
		predicted_age = sum/counter
		return predicted_age

	def age_prediction(self):
		try:
			actual_age = self.filename.split('/')[-1].split('_')[0]
			img = cv2.imread(self.filename) 
			predict_age = self.predict_with_preprocessing(img)
			self.queue = queue.Queue()
			ThreadedTask(self.queue).start()
			self.root.after(10, self.process_queue)
			# print("Approximate face Age is "+str(age))
			self.label_info['text'] = 'Status: '
			self.label_predicted_age_val['text'] = str(predict_age)
			try:
				int(actual_age)
				self.label_actual_age_val['text'] = str(actual_age)
				self.label_error_val['text'] = str(round(float(predict_age)-float(actual_age),2))
			except:
				self.label_actual_age_val['text'] = 'NA'
				self.label_error_val['text'] = 'NA'
		except:
			self.label_info['text'] = 'Status: \nSelect an image\nbefore procedding further.'


	def selectImage(self):
		self.root.filename =  filedialog.askopenfilename(initialdir = "C:/python35/scripts/env/upload-image", title = "Select file", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
		self.filename = self.root.filename
		try:
			self.label_info['text'] = 'Status: '
			image = Image.open(self.filename)
			baseheight = 200
			hpercent = (baseheight / float(image.size[1]))
			wsize = int((float(image.size[0]) * float(hpercent)))
			image = image.resize((wsize, baseheight), Image.ANTIALIAS)
			image = ImageTk.PhotoImage(image)
			self.panel.configure(image=image)
			self.panel.image = image
		except:
			self.label_info['text'] = 'Status: \nUnable to get file name.'

	def make_model(self):
		global filters,conv, img_w, img_h, img_channels, pool
		self.model = Sequential()
		self.model.add(Conv2D(filters, (conv, conv), padding="same", input_shape=(img_w, img_h, img_channels)))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization(axis=chanDim))
		self.model.add(MaxPooling2D(pool_size=(pool, pool)))
		self.model.add(Dropout(0.2))

		pool -= 1
		filters *= 2
		self.model.add(Conv2D(filters, (conv, conv), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization(axis=chanDim))
		self.model.add(Conv2D(filters, (conv, conv), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization(axis=chanDim))
		self.model.add(MaxPooling2D(pool_size=(pool, pool)))
		self.model.add(Dropout(0.2))

		filters *= 2
		self.model.add(Conv2D(filters, (conv, conv), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization(axis=chanDim))
		self.model.add(Conv2D(filters, (conv, conv), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization(axis=chanDim))
		self.model.add(MaxPooling2D(pool_size=(pool, pool)))
		self.model.add(Dropout(0.2))

		filters *= 2
		self.model.add(Conv2D(filters, (conv, conv), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization(axis=chanDim))
		self.model.add(Conv2D(filters, (conv, conv), padding="same"))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization(axis=chanDim))
		self.model.add(MaxPooling2D(pool_size=(pool, pool)))
		self.model.add(Dropout(0.2))

		filters *= 4
		self.model.add(Flatten())
		self.model.add(Dense(filters))
		self.model.add(Activation("relu"))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(classes))
		self.model.add(Activation("softmax"))        

	def process_queue(self):
		try:
			self.queue.get(0)
		except queue.Empty:
			self.root.after(10, self.process_queue)

class ThreadedTask(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
    def run(self):
        time.sleep(5)  # Simulate long running process
        self.queue.put("Task finished")

if __name__=='__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
	root = Tk()
	main_ui = GUI(root)