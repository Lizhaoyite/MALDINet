import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.layers import MaxPool2D, GlobalMaxPool2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Concatenate,Flatten, Dense, Dropout
import pandas as pd
import numpy as np
import os
import cv2
from joblib import load, dump
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score,precision_score
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold,KFold,RepeatedKFold,RepeatedStratifiedKFold, train_test_split
from aggmap import AggMap, AggMapNet

def Inception(inputs, units = 8, strides = 1):
    """
    naive google inception block
    """
    x1 = Conv2D(units, 5, padding='same', activation = 'relu', strides = strides)(inputs)
    x2 = Conv2D(units, 3, padding='same', activation = 'relu', strides = strides)(inputs)
    x3 = Conv2D(units, 1, padding='same', activation = 'relu', strides = strides)(inputs)
    outputs = Concatenate()([x1, x2, x3])
    outputs = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.8)(outputs)    
    return outputs

def MK(inputs, units = [12, 24, 48], strides = 1):
    """
    multi kernel block
    """
    input_shape = inputs.shape[1] if inputs.shape[1] < inputs.shape[2] else inputs.shape[2]  # 获取输入图像尺寸
    if input_shape > 7:
        x1 = Conv2D(units[0], 7, padding='same', activation='relu', strides=strides)(inputs)
        x2 = Conv2D(units[1], 5, padding='same', activation='relu', strides=strides)(inputs)
        x3 = Conv2D(units[2], 3, padding='same', activation='relu', strides=strides)(inputs)
    elif input_shape > 5:
        # 使用两个3x3和一个5x5卷积核
        x1 = Conv2D(units[0], 5, padding='same', activation='relu', strides=strides)(inputs)
        x2 = Conv2D(units[1], 3, padding='same', activation='relu', strides=strides)(inputs)
        x3 = Conv2D(units[2], 3, padding='same', activation='relu', strides=strides)(inputs)
    
    else:
        # 使用三个3x3卷积核
        x1 = Conv2D(units[0], 3, padding='same', activation='relu', strides=strides)(inputs)
        x2 = Conv2D(units[1], 3, padding='same', activation='relu', strides=strides)(inputs)
        x3 = Conv2D(units[2], 3, padding='same', activation='relu', strides=strides)(inputs)
    outputs = Concatenate()([x1, x2, x3])
    outputs = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.8)(outputs)
    return outputs

def DK(inputs, units = [12, 24], strides = 1):
    """
    double kernel block
    """
    input_shape = inputs.shape[1] if inputs.shape[1] < inputs.shape[2] else inputs.shape[2]  # 获取输入图像尺寸
    if input_shape > 7:
        x1 = Conv2D(units[0], 5, padding='same', activation='relu', strides=strides)(inputs)
        x2 = Conv2D(units[1], 3, padding='same', activation='relu', strides=strides)(inputs)
    else:
        x1 = Conv2D(units[0], 3, padding='same', activation='relu', strides=strides)(inputs)
        x2 = Conv2D(units[1], 3, padding='same', activation='relu', strides=strides)(inputs)
    outputs = Concatenate()([x1, x2])
    outputs = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.8)(outputs)
    return outputs

def _MALDINet(input_shape,  
               n_outputs = 1,
               n_inception = 2,
               kernel_block = 'MK',
               units = [12, 24, 48],
               dense_layers = [256, 128, 64, 32], 
               dense_avf = 'relu', 
               dropout = 0,
               last_avf = 'softmax'):

    """
    parameters
    ----------------------
    input_shape: w, h, c
    n_outputs: output units
    n_inception: number of the inception layers
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    dropout: dropout of the dense layers
    """
    tf.keras.backend.clear_session()
    assert len(input_shape) == 3
    inputs = Input(input_shape)
    
    if kernel_block == 'MK':
        incept = MK(inputs=inputs, units=units)  
    elif kernel_block == 'DK':
        incept = DK(inputs=inputs, units=units)
    else : incept = Conv2D(units[0], int(kernel_block), padding = 'same', activation='relu', strides = 1)(inputs)

    for i in range(n_inception):
        incept = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(incept) #p1
        incept = Inception(incept, strides = 1, units = 32*(2**i))

    #flatten
    x = GlobalMaxPool2D()(incept)
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)
        if dropout:
            x = Dropout(rate = dropout)(x)

    #last layer
    outputs = Dense(n_outputs,activation = last_avf)(x)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    
    return model