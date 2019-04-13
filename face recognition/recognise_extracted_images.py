from keras.models import load_model
import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

class_pred = load_model('Face_Recognition.h5')

class_pred.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



path = 'Extracted/'             #images to be recognised are in Extracted folder
all_files = os.listdir(path)
for file in all_files:
    img = cv2.imread(path+file)
    img = cv2.resize(img,(64,64))
    img = np.reshape(img,[1,64,64,3])
    classes = class_pred.predict_classes(img)
    print(file,classes)
    

