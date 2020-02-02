#importing libraries
import csv
import os
import cv2
import sklearn
import numpy as np
from math import ceil
from random import shuffle
from keras.models import Sequential
from keras.layers import Flatten,Dense,Conv2D
from keras.layers import Lambda,Dropout
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l1
from keras.layers import Cropping2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#collecting data from driving_log.csv file 
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
#split data 
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


#use a generator to load data and preprocess it on the fly
def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction=0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
                
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                name_left='./data/IMG/'+batch_sample[1].split('/')[-1]
                name_right='./data/IMG/'+batch_sample[2].split('/')[-1]
                center_image =cv2.imread(name)
                left_image =cv2.imread(name_left)
                right_image =cv2.imread(name_right)
                #print(center_image.shape)
                center_image=cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                #flipp the center image 
                #image_flipped = np.fliplr( center_image)
               
                
                left_image=cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image=cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                #add flipped steering angle
                #measurement_flipped = -center_angle
                left_angle=center_angle+correction
                right_angle=center_angle-correction
                
                images.append(center_image)
                #images.append(image_flipped)
                images.append(left_image)
                images.append(right_image)
                
                angles.append(center_angle)
                #angles.append(measurement_flipped)
                angles.append(left_angle)
                angles.append(right_angle)
              
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
# Set our batch size
batch_size=32
ch, row, col = 3, 80, 320 
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#define the model
model=Sequential()
#normalization 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#cropping images
model.add(Cropping2D(cropping=((50,25),(0,0)))) 
#nvidia model 
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Flatten())
model.add(Dense(100,activation='elu'))
#add a regularization method
model.add(Dropout(0.25))
model.add(Dense(50,activation='elu'))
model.add(Dense(10,activation='elu'))
model.add(Dense(1))

#model compiling with Adam optimizer
model.compile(loss='mse',optimizer='adam')

#store the loss values during the training
history_object=model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
           epochs=5, verbose=1)
#save the model 
model.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('model2_flipped.png')