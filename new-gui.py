from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as ttk
import time
import threading
import cv2
import os

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras import backend as k
from keras.models import load_model
import tensorflow as tf

graph = tf.get_default_graph()

img_w = 75
img_h = 75
weight_dir = os.path.join(os.getcwd(), 'weights/')
model_name = 'face_model.h5'
model_dir = os.path.join(os.getcwd(), 'models/')

class GUI:
	def __init__(self,root, model):
		self.filename = None
		self.root = root
		self.panel = None
		self.root.geometry('680x400')
		self.root.resizable(0, 0)
		self.model = model

		self.label_info = Label(self.root, text='Status: \nWelcome!!')
		self.label_info.grid(column=1,row=3, padx=5, pady=3, sticky=S)

		self.btn_select_image = Button(self.root, text="Select Image", command = self.selectImage)
		self.btn_select_image.grid(column=4,row=3, padx=3, pady=3, sticky=S)

		self.btn_predict = Button(self.root, text="Predict Age", command = self.threaded_age_prediction)
		self.btn_predict.grid(column=5,row=3, padx=3, pady=3, sticky=S)

		self.btn_quit = Button(self.root, text='Quit', command = lambda : self.root.destroy())
		self.btn_quit.grid(column=6,row=3, padx=3, pady=3, sticky=S)

		try:
			self.filename = 'testing-image/sample-image.jpg'
			image = Image.open(self.filename)
			baseheight = 200
			hpercent = (baseheight / float(image.size[1]))
			wsize = int((float(image.size[0]) * float(hpercent)))
			image = image.resize((wsize, baseheight), Image.ANTIALIAS)
			image = ImageTk.PhotoImage(image)
			self.panel = Label(image=image)
			self.panel.image = image
		except Exception as e:
			print('ERROR: ',e)
			pass
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

		self.root.grid_columnconfigure(6, minsize=80)
		self.root.grid_columnconfigure(5, minsize=80)
		self.root.grid_columnconfigure(4, minsize=80)
		self.root.grid_columnconfigure(3, minsize=100)
		self.root.grid_rowconfigure(3, minsize=80)
		self.root.grid_columnconfigure(2, minsize=100)
		self.root.grid_rowconfigure(2, minsize=80)
		self.root.grid_columnconfigure(1, minsize=100)
		self.root.grid_rowconfigure(1, minsize=140)

		self.root.wm_title("Age Predictor")
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
		except:
			print('Unable to find face')
			print('using whole picture')
			gray = cv2.resize(gray, (img_w,img_h))
			gray = np.expand_dims(gray, axis=2)
			gray = np.array([gray])/255.0

		sum=0.0
		counter=0.0
		with graph.as_default():
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
			self.label_info['text'] = 'Status: \nPredicting...'
			self.btn_predict.config(state = DISABLED)
			predict_age = self.predict_with_preprocessing(img)
			# print("Approximate face Age is "+str(age))
			self.label_predicted_age_val['text'] = str(predict_age)
			try:
				int(actual_age)
				self.label_actual_age_val['text'] = str(actual_age)
				self.label_error_val['text'] = str(round(float(predict_age)-float(actual_age),2))
			except:
				self.label_actual_age_val['text'] = 'NA'
				self.label_error_val['text'] = 'NA'
			self.label_info['text'] = 'Status: \nDone!!!'
			self.btn_predict.config(state = NORMAL)
		except:
			self.label_info['text'] = 'Status: \nChoose a file before\npressing predict button'
			return 
		finally:
			self.btn_predict.config(state = NORMAL)

	def selectImage(self):
		self.filename =  filedialog.askopenfilename(initialdir = "C:/python35/scripts/env/upload-image", title = "Select file", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
		try:
			self.label_info['text'] = 'Status: \nFile Selected'
			self.btn_predict.config(state = NORMAL)
			image = Image.open(self.filename)
			baseheight = 210
			hpercent = (baseheight / float(image.size[1]))
			wsize = int((float(image.size[0]) * float(hpercent)))
			image = image.resize((wsize, baseheight), Image.ANTIALIAS)
			image = ImageTk.PhotoImage(image)
			self.panel.configure(image=image)
			self.panel.image = image
			w = float(680/400)*wsize*1.0
			h = float(400/200)*baseheight*1.0
			w = int(w)
			h = int(h)
			wxh = str(max(w,680))+'x'+str(max(h,400))
			# print(wxh)
			print(self.filename)
			self.root.geometry(wxh)
		except:
			self.label_info['text'] = 'Status: \n'

	def threaded_age_prediction(self):
		self.thread = threading.Thread(target = self.age_prediction)
		self.thread.start()

if __name__=='__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
	root = Tk()
	model = load_model(model_dir+model_name)
	main_ui = GUI(root, model)