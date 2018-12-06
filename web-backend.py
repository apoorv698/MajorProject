# importing required Modules

from flask import Flask, render_template, request, redirect, url_for
import base64
import re
import numpy as np
from io import BytesIO
from tkinter import *
from PIL import Image, ImageTk
import time
import threading
import cv2
import os

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as k
from keras.models import load_model

# Flask constructor call 
app = Flask(__name__)

# Constants Declaration
img_w = 75
img_h = 75
predictedAge='NA'
model_name = 'face_model.h5'

# storing different path
weight_dir = os.path.join(os.getcwd(), 'weights/')
model_dir = os.path.join(os.getcwd(), 'models/')

# Returns the default graph for the current thread
graph = tf.get_default_graph()

# making the function for '/result' address for 'get' request when redirection comes from 'index()' 
@app.route('/result' , methods=['GET'])
def result():
	global predictedAge
	# logging predicted age
	print(predictedAge)

	return render_template('result.html', predictedAge=predictedAge)

# providing the backend for '/' address for both 'get' and 'post' request 
@app.route('/', methods=['POST','GET'])
def index():
	# logging request type either 'get' or 'post'
	print(request.method)

	# if it is a 'post' request 
	if request.method == 'POST':
		# using default graph in this thread
		with graph.as_default():
			global predictedAge

			# getting image from html page using javascript
			image_b64 = request.form['image']
			image_b64 = image_b64.split(',')[1]
			# print(image_b64[:100])
			# converting  base64 text into image
			sbuf = BytesIO()
			sbuf.write(base64.b64decode(image_b64))
			pimg = Image.open(sbuf)
			img = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

			global weight_dir, img_w, img_h
			
			# Processing the 'img' image file
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			gray = cv2.equalizeHist(gray)

			# Loading 'haarcascade_frontalface_default.xml'  to find faces 
			face_cascade = cv2.CascadeClassifier('C:/Python35/Scripts/env/haarcascade_frontalface_default.xml')
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)

			# finding region of image that contain face			
			roi = None
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

			# Loading the weights and predicting the class ie the age of the preson
			for wt in os.listdir(weight_dir):
				counter+=1.0
				model.load_weights(weight_dir+wt)
				print("wt: ",wt)
				try:
					ynew = model.predict_classes(gray_img)
				except:
					ynew = model.predict_classes(gray)
				sum+=ynew[0]

			# finding predicted age
			predictedAge = sum/counter
			
			# logging predicted age
			print('predict_age=', predictedAge)
			
			# redirecting to '/result' address
			return redirect(url_for('result'))

	# if it is a 'get' request 
	else:
		
		# loading the default page ie 'index.html' stored in templates folder
		return render_template('index.html')


if __name__ =="__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

	# Loading the Model
	model = load_model(model_dir+model_name)
	print('model prepared')

	# running the app
	app.run(debug=False,port=8080)