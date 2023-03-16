#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 23:22:26 2023

@author: ben
"""

import sys
sys.path.append("../Utils")
from plot import plot
import numpy as np
from tensorflow import keras
from models import mnist_model




def mnist_example():
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)
    
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = mnist_model(input_shape,num_classes)
    
    batch_size = 256
    epochs = 5
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    plot(x=[],
         y=[history.history['loss'],history.history['val_loss']],
         xlabel="Epochen",
         ylabel="Fehler",
         #ylim=(-5,7),
         #xlim=(0,70),
         figsize_x=4.5,
         legend=["Training","Validierung"],
         legend_loc="upper right")
    plot(x=[],
         y=[history.history['accuracy'],history.history['val_accuracy']],
         xlabel="Epochen",
         ylabel="Genauigkeit",
         #ylim=(-5,7),
         #xlim=(0,70),
         figsize_x=4.5,
         legend=["Training","Validierung"],
         legend_loc="lower right")

if __name__ == "__main__":
    mnist_example()