# All Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from tqdm import tqdm

# Setup Keras
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import GaussianNoise
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Here this configuration was needed to use the GPU, probably because o CUDA and CUDnn
# otherwise it wouldnt run
config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )

sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# Read the driving.csv with all the collected data
drive_csv = pd.read_csv('driving_log.csv', header=None)

# Dropping and working with the dataframe for a better and easier future generator
drive_csv.drop(columns=[4,5,6], inplace=True)

# Dealing with the names of the files
drive_csv['center'] = drive_csv[0].apply(lambda x: x.split('/')[-1])
drive_csv['left'] = drive_csv[1].apply(lambda x: x.split('/')[-1])
drive_csv['right'] = drive_csv[2].apply(lambda x: x.split('/')[-1])

# Generating the dataframe for the generator
drive_dict = pd.DataFrame()

for i in tqdm(range(len(drive_csv))):

    # Storing the data
    images = []
    measurements = []
    
    # Get the center measurement for angle correction for the right and left image
    measurement_center = float(drive_csv.iloc[i, 3])
    
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = measurement_center + correction
    steering_right = measurement_center - correction

    # Appending all data
    measurements.append(measurement_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
    images.append(drive_csv.iloc[i, 4])
    images.append(drive_csv.iloc[i, 5])
    images.append(drive_csv.iloc[i, 6])
    
    # Storing in a dataframe for a cleaner generator (batches)
    for j in range(3):
        drive_dict = drive_dict.append({'images': images[j], 'angle': measurements[j]}, 
                                        ignore_index=True)

# Example code from Udacity to get the samples for the generator
samples = []
for line in drive_dict.values:
    samples.append(line)

# Using sklearn to split the data in train and validation, chose a split of 25% for Validation
train_samples, validation_samples = train_test_split(samples, test_size=0.25)

# Creating the generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for batch_sample in batch_samples:
                measurement_center = float(batch_sample[1])
                
                # Get the image and convert to RGB
                image_center = cv2.imread('./IMG/' + batch_sample[0])
                image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)
                images.append(image_center)
                measurements.append(measurement_center)


            # Transform into array
            X_train = np.array(images)
            y_train = np.array(measurements)                
            
            yield shuffle(X_train, y_train)

# Model Architecture
# Inspired on the NVIDIA model, modified the fully connected layer
model = Sequential()

# Lambda layer for normalization, GaussianNoise for better generalization and
# the Cropping for the better ROI
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(GaussianNoise(0.1))
model.add(Cropping2D(cropping=((70,25), (0,0))))

#Layers just like NVIDIA model
model.add(Conv2D(24, (5,5), activation='relu'))

# Added a MaxPooling on these next layer for a smaller model
# The performance was better with same Mean Squared Error
model.add(Conv2D(36, (5,5), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(48, (5,5), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())

# Fully Connected, made it a little smaller from NVIDIA
model.add(Flatten())

# Added DropOut on the fully connected layer for better regularization
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output of the model, single neuron for angle prediction
model.add(Dense(1))

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# I chose a lower lr for the Adam, instead of the 1e-3, made a better convergence
optim = Adam(lr=0.0001)

# Model compiled with the MSE error for the regression task
model.compile(loss='mse', optimizer=optim, metrics=['mse'])

# Model training
model.fit_generator(train_generator,
                    steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=np.ceil(len(validation_samples)/batch_size),
                    epochs=7, verbose=1)

# After the training save the model
model.save('model_trained.h5')