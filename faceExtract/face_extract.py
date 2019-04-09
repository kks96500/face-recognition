#importing the library
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from random import randint
import sys
import os
import face_recognition
import glob

#taking photos from camera
capture_duration=0.3
start_time=time.time()
cap=cv.VideoCapture(0)
count=0
while(int(time.time() - start_time) < capture_duration):
	#capture frame by frame 
	ret,frame=cap.read()
	gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
	cv.imwrite("frame%d.jpg"%count,gray)
	count+=1
	#cv.imshow('frame',gray)
cap.release()
#cv.destroyAllWindows()
      
#detecting and extracting the faces
CASCADE="haarcascade_frontalface_alt.xml"
FACE_CASCADE=cv.CascadeClassifier(CASCADE)

def detect_faces(image_path):

	image=cv.imread(image_path)
	image_grey=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

	faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)

    #Finding Faces
	for x,y,w,h in faces:
	    sub_img=image[y-10:y+h+10,x-10:x+w+10]
	    os.chdir("Extracted")
	    cv.imwrite(str(randint(0,10000))+".jpg",sub_img)
	    os.chdir("../")

	if (cv.waitKey(0) & 0xFF == ord('q')) or (cv.waitKey(0) & 0xFF == ord('Q')):
		cv.destroyAllWindows()

if __name__ == "__main__":
	
	#Creating folder for extracted image
	if not "Extracted" in os.listdir("."):
		os.mkdir("Extracted")
    
	for img in glob.glob('*.jpg'):
			detect_faces(img)
#Removing the images taken by webcam
for i in glob.glob('*.jpg'):
	os.remove(i)


































