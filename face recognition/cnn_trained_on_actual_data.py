#importing keras library and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#part 1
#Intialsing the CNN
classifier=Sequential()

#step 1-convolution 
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#step 2-maxpooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3-Flattening
classifier.add(Flatten())

#step 4 -full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 6, activation = 'softmax'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#part 2 fiiting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Dataset/Training_Set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'Dataset/Test_Set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=391,
        epochs=5,
        validation_data=test_set,
        validation_steps=76)


classifier.save('Face_Recognition.h5')



