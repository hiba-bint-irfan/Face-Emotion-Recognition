# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:57:54 2023

@author: Irfan
"""


import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Initialize image data generator with rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'Dataset/training_set',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'Dataset/test_set',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

emotion_cnn_model = Sequential()

emotion_cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_cnn_model.add(Dropout(0.25))

emotion_cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_cnn_model.add(Dropout(0.25))

emotion_cnn_model.add(Flatten())
emotion_cnn_model.add(Dense(1024, activation='relu'))
emotion_cnn_model.add(Dropout(0.5))
emotion_cnn_model.add(Dense(6, activation='softmax'))

emotion_cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


# cnn.fit(x=training_set,validation_data=test_set,epochs=30)
# Train the neural network/model
emotion_cnn_model.fit(
        x=training_set,
        steps_per_epoch=19638 // 64,
        epochs=50,
        validation_data=test_set,
        validation_steps=7879 // 64)

# save model structure in jason file
model_json = emotion_cnn_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_cnn_model.save('emotion_model.h5')
