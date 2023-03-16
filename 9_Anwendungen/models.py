#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:43:30 2023

@author: ben
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2DTranspose,Dropout
from tensorflow.keras.regularizers import l2


def pyramid_maps(input_net,filter_size):
    net1 = Conv2D(filters=16, kernel_size=(1,1), padding='same',kernel_regularizer=l2(0.0005))(input_net)
    net1 = BatchNormalization()(net1)
    net1 = Activation('relu')(net1)
    net1=Dropout(0.3)(net1)
    
    net2 = Conv2D(filters=32, kernel_size=(3,3), padding='same',kernel_regularizer=l2(0.0005))(input_net)
    net2 = BatchNormalization()(net2)
    net2 = Activation('relu')(net2)
    net2 = Dropout(0.3)(net2)
    
    net3 = Conv2D(filters=32, kernel_size=(5,5), padding='same',kernel_regularizer=l2(0.0005))(input_net)
    net3 = BatchNormalization()(net3)
    net3 = Activation('relu')(net3)
    net3 = Dropout(0.3)(net3)
    
    out = concatenate([net1,net2,net3], axis=-1)#skip connection
    return out




def pspnet(input_shape,
                  n_labels,
                  num_filters=64,
                  output_mode="softmax"):
    
    #input
    input_tensor = Input(shape=input_shape, name='input_tensor')
    
    ####################################
    # encoder (contracting path)
    ####################################
    #encoder block 0
    e0 = pyramid_maps(input_tensor,32)
    
    #encoder block 1
    e1 = AveragePooling2D((2, 2))(e0)
    e1 = pyramid_maps(e1,32)
    
    #encoder block 2
    e2 = AveragePooling2D((2, 2))(e1)
    e2 = pyramid_maps(e2,32)
    
    ####################################
    # decoder (expansive path)
    ####################################
    
    #decoder block 1
    d1 = UpSampling2D((2, 2),)(e2)
    d1 = pyramid_maps(d1,32)
    d1 = concatenate([e1,d1], axis=-1)#skip connection
    
    #decoder block 0
    d0 = UpSampling2D((2, 2),)(d1)
    d0 = pyramid_maps(d0,32)
    d0 = concatenate([e0,d0], axis=-1)#skip connection
    
    #output
    out_class = Conv2D(n_labels, (1, 1), padding='same')(d0)
    out_class = Activation(output_mode,name='output')(out_class)
    
    pspnet = Model(inputs=input_tensor, outputs=out_class)
    
    return pspnet

def mnist_model(input_shape,n_labels):
    """
    Creates a convolutional neural network (CNN) 
    for classifying images of handwritten digits 
    from the MNIST dataset.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the input images 
        in the format (height, width, 
        channels).
    n_labels : int
        Number of classes to classify.
    
    Returns:
    --------
    keras.Model:    
        A compiled keras model for training 
        and testing.
    """
    input_tensor = Input(shape=input_shape, name='input_tensor')
    x = Conv2D(filters=32, kernel_size=(3,3), padding='same',kernel_regularizer=l2(0.0005))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(filters=64, kernel_size=(3,3), padding='same',kernel_regularizer=l2(0.0005))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = AveragePooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = Dropout(0.3)(x)
    output_tensor = layers.Dense(n_labels, activation="softmax")(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()
    return model


if __name__ == "__main__":
    pass