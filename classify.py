from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from attention_layer import *
import tensorflow as tf
import numpy as np
import cv2
import os
import MSA


def main(x_train,y_train,path):

    """momentum-attention based EfficientNet classification"""
    input_shape=(200,200,1)
    
    model = Sequential()
    model.add(Conv2D(3, (3, 3), padding='same', activation='relu',input_shape=input_shape))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(24, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Attention()) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(24, (5, 5), activation='relu'))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(Conv2D(176, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256,activation = 'relu'))
    model.add(Dense(4,activation = 'softmax'))
    opt = tf.keras.optimizers.SGD(learning_rate=MSA.optimize(0.1), momentum=0.5)
    model.compile(optimizer="adam", loss = 'categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    model.fit(x_train,y_train,epochs=100,verbose=True,batch_size=8)
    
    model.save(path)
    
    

