# importing required Modules
# using tkinter for GUI

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

# Returns the default graph for the current thread
graph = tf.get_default_graph()

# Constants Declaration
img_w = 75
img_h = 75
model_name = 'face_model.h5'

# storing different path
weight_dir = os.path.join(os.getcwd(), 'wieghts/')
model_dir = os.path.join(os.getcwd(), 'models/')

class GUI:
	# constructor for the class 'GUI'
	def __init__(self,root, model):
		self.filename = None
		self.root = root
		self.panel = None
		self.root.geometry('680x400')
		self.root.resizable(0, 0)
		self.model = model

		# inialization of GUI labels, buttons and panel to display selected image

		self.label_info = Label(self.root, text='Status: \nWelcome!!')
		self.label_info.grid(column=1,row=3, padx=5, pady=3, sticky=S)

		self.btn_select_image = Button(self.root, text="Select Image", command = self.selectImage)
		self.btn_select_image.grid(column=4,row=3, padx=3, pady=3, sticky=S)

		self.btn_predict = Button(self.root, text="Predict Age", command = self.threaded_age_prediction)
		self.btn_predict.grid(column=5,row=3, padx=3, pady=3, sticky=S)

		self.btn_quit = Button(self.root, text='Quit', command = lambda : self.root.destroy())
		self.btn_quit.grid(column=6,row=3, padx=3, pady=3, sticky=S)

		try:
			# displaying sample image
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

		# using grid layout option
		self.panel.grid(column=3, row=1, columnspan=2)

		# setting label configurations in GUI
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

		# configuring grid layout parameters

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

	# 'predict_with_preprocessing' used for predicting age from the image

	def predict_with_preprocessing(self, img):
		global weight_dir, img_w, img_h

		# Processing the 'img' image file

		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)

		# Loading 'haarcascade_frontalface_default.xml'  to find faces 
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		roi = None

		# finding region of image that contain face			
		for (x,y,w,h) in faces: 
			roi = gray[y:y+h, x:x+w]

		# checking if roi contains some face or not
        # ie. if some face have been detected in the image

		try:
			# running when face is found
			print('using face only')
			gray_img = cv2.resize(roi, (img_w,img_h))
			gray_img = np.expand_dims(gray_img, axis=2)
			gray_img = np.array([gray_img])/255.0
		except:
			# running when face is not found
			print('Unable to find face')
			print('using whole picture')
			gray = cv2.resize(gray, (img_w,img_h))
			gray = np.expand_dims(gray, axis=2)
			gray = np.array([gray])/255.0

		sum=0.0
		counter=0.0

		# using default graph in this thread
		with graph.as_default():
			# Loading the weights and predicting the class ie the age of the preson
			for wt in os.listdir(weight_dir):
				counter+=1.0
				self.model.load_weights(weight_dir+wt)
				try:
					ynew = self.model.predict_classes(gray_img)
				except:
					ynew = self.model.predict_classes(gray)
				sum+=ynew[0]
		
		# finding predicted age
		print(sum, counter)
		predicted_age = sum/counter
		
		return predicted_age


	def age_prediction(self):
		try:
			# getting actual_age from file name
			actual_age = self.filename.split('/')[-1].split('_')[0]
			img = cv2.imread(self.filename) 
			self.label_info['text'] = 'Status: \nPredicting...'
			
			#changing 'Predict' button status to diabled to avoid repeated clicking
			self.btn_predict.config(state = DISABLED)

			predict_age = self.predict_with_preprocessing(img)

			# chaning the label content with the age
			self.label_predicted_age_val['text'] = str(predict_age)

			try:
				# image contanins the age value in it
				int(actual_age)
				self.label_actual_age_val['text'] = str(actual_age)
				self.label_error_val['text'] = str(round(float(predict_age)-float(actual_age),2))
			except:
				# if filename does not have age value then 'NA' is displayed
				self.label_actual_age_val['text'] = 'NA'
				self.label_error_val['text'] = 'NA'

			# status label is chenged
			self.label_info['text'] = 'Status: \nDone!!!'

			# button status is changed
			self.btn_predict.config(state = NORMAL)
		except:
			self.label_info['text'] = 'Status: \nChoose a file before\npressing predict button'
			return 
		finally:
			self.btn_predict.config(state = NORMAL)

	def selectImage(self):
		# setting default directory
		self.filename =  filedialog.askopenfilename(initialdir = "C:/Users/apoov/Desktop/real-test-image", title = "Select file", filetypes = (("all files","*.*"),("jpeg files","*.jpg")))
		try:
			# changing status 
			self.label_info['text'] = 'Status: \nFile Selected'
			self.btn_predict.config(state = NORMAL)
			image = Image.open(self.filename)

			# resizing image for diplay purposes			
			baseheight = 210
			hpercent = (baseheight / float(image.size[1]))
			wsize = int((float(image.size[0]) * float(hpercent)))
			image = image.resize((wsize, baseheight), Image.ANTIALIAS)
			image = ImageTk.PhotoImage(image)

			self.panel.configure(image=image)
			self.panel.image = image
			
			# setting height and widht
			w = float(680/400)*wsize*1.0
			h = float(400/200)*baseheight*1.0
			w = int(w)
			h = int(h)
			wxh = str(max(w,680))+'x'+str(max(h,400))
			
			# logging file name
			print(self.filename)
			self.root.geometry(wxh)
		except:
			self.label_info['text'] = 'Status: \n'

	# making thread to avoid freezing of window when prediction is being done
	def threaded_age_prediction(self):
		self.thread = threading.Thread(target = self.age_prediction)
		self.thread.start()

if __name__=='__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

	# initialization of tkinter 
	root = Tk()

	# loading model
	model = load_model(model_dir+model_name)
	
	# making UI object using tkinter init
	main_ui = GUI(root, model)