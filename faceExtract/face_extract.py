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
CASCADE="/home/kks96500/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
FACE_CASCADE=cv.CascadeClassifier(CASCADE)

def detect_faces(image_path):

	image=cv.imread(image_path)
	image_grey=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

	faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)

	for x,y,w,h in faces:
	    sub_img=image[y-10:y+h+10,x-10:x+w+10]
	    os.chdir("Extracted")
	    cv.imwrite(str(randint(0,10000))+".jpg",sub_img)
	    os.chdir("../")
	    #cv.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

	#cv.imshow("Faces Found",image)
	if (cv.waitKey(0) & 0xFF == ord('q')) or (cv.waitKey(0) & 0xFF == ord('Q')):
		cv.destroyAllWindows()

if __name__ == "__main__":
	
	if not "Extracted" in os.listdir("."):
		os.mkdir("Extracted")
	for img in glob.glob('*.jpg'):
			detect_faces(img)
for i in glob.glob('*.jpg'):
	os.remove(i)



'''#recognising faces

# make a list of all the available images
images = os.listdir('images')
extract=os.listdir('Extracted')
# load your image
for img in extract:
	image_to_be_matched = face_recognition.load_image_file(img)

	# encoded the loaded image into a feature vector
	image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]
	# iterate over each image
	for image in images:
    		# load the image
    		current_image = face_recognition.load_image_file("images/" + image)
    		# encode the loaded image into a feature vector
    		current_image_encoded = face_recognition.face_encodings(current_image)[0]
    		#match your image with the image and check if it matches
    		result = face_recognition.compare_faces([image_to_be_matched_encoded], current_image_encoded)
    		#check if it was a match
    		if result[0] == True:
        		print("Matched: "+image)
    		else:
        		print("Not matched: " + image)'''






























