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


app = Flask(__name__)

img_channels = 1
img_w = 80
img_h = 80
batch_size = 64
classes = 31
epoch = 20
filters = 32
pool = 3
conv = 3
chanDim = -1

data_dir = os.path.join(os.getcwd(), 'data-model-2')
weight_dir = os.path.join(os.getcwd(), 'weights-model-2/') 
model_name = 'face_model.h5'
model_dir = os.path.join(os.getcwd(), 'models-model-2/')

predictedAge='NA'
graph = tf.get_default_graph()


@app.route('/result' , methods=['GET'])
def result():
	global predictedAge
	print(predictedAge)
	return render_template('result.html',predictedAge=predictedAge)


@app.route('/', methods=['POST','GET'])
def index():
	print(request.method)
	if request.method == 'POST':
		with graph.as_default():
			global predictedAge
			# print("INSIDE POST")
			# print(request.form['number'])
			image_b64 = request.form['image']
			image_b64 = image_b64.split(',')[1]
			print(image_b64[:100])
			sbuf = BytesIO()
			sbuf.write(base64.b64decode(image_b64))
			pimg = Image.open(sbuf)
			image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
			# print('image produced')
			# print(image.shape)
			#cv2.imread('captured image', (image))
			#cv2.waitKey(0)
			global weight_dir, img_w, img_h
			img = image
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			face_cascade = cv2.CascadeClassifier('C:/Python35/Scripts/env/haarcascade_frontalface_default.xml')
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			# print('displaying image')
			
			roi = None
			for (x,y,w,h) in faces: 
				roi = gray[y:y+h, x:x+w]
			try:
				print('using face only')
				gray_img = cv2.resize(roi, (img_w,img_h))
				gray_img = np.expand_dims(gray_img, axis=2)
				gray_img = np.array([gray_img])/255.0
				#cv2.imshow('face', (gray_img))
			except:
				print('Unable to find face')
				print('using whole picture')
				gray = cv2.resize(gray, (img_w,img_h))
				gray = np.expand_dims(gray, axis=2)
				gray = np.array([gray])/255.0
				print(gray.shape)
				#cv2.imshow('face', (gray))
			
			predicted_age = 0
			
			sum=0.0
			counter=0.0
			try:
				for wt in os.listdir(weight_dir):
					counter+=1.0
					model.load_weights(weight_dir+wt)
					print("wt: ",wt)
					try:
						ynew = model.predict_classes(gray_img)
					except:
						ynew = model.predict_classes(gray)
					print(ynew[0])
					sum+=ynew[0]*3 - 1.5
			except Exception as e:
				print('line 217 ',e)
			predicted_age = sum/counter
		
			predictedAge = predicted_age
			# predictedAge = 22
			print('predict_age=', predictedAge)
			return redirect(url_for('result'))
	else:
		return render_template('index.html')


if __name__ =="__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
	
	model = load_model(model_dir+model_name)
	print('model prepared')

	app.run(debug=False,port=8080)