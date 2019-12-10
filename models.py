#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:34:48 2019

@author: henric
"""
import tensorflow.keras as keras
from tensorflow.keras import layers 
from tensorflow.keras import regularizers

PLANES = 8
MOVES = 361

def get_model_baseline():
    input = keras.Input(shape=(19, 19, PLANES), name='board')
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(input)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    policy_head = layers.Conv2D(1, 3, activation='relu', padding='same')(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(MOVES, activation='softmax', name='policy')(policy_head)
    value_head = layers.Flatten()(x)
    value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)
    
    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
    return model

def get_model_week2(convol_size=128):
    input = keras.Input(shape=(19, 19, PLANES), name='board')
    x = layers.Conv2D(convol_size, 3, padding='valid')(input)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    
    y = layers.Conv2D(convol_size, 3, padding='same')(x)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(convol_size, 3, padding='same')(y)
    y = layers.BatchNormalization(axis=-1)(y)
    x = layers.Add()([x,y])
    x = layers.Activation('relu')(x)
    
    y = layers.Conv2D(convol_size, 3, padding='same')(x)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(convol_size, 3, padding='same')(y)
    y = layers.BatchNormalization(axis=-1)(y)
    x = layers.Add()([x,y])
    x = layers.Activation('relu')(x)
    
    policy_head = layers.Conv2D(2, 1)(x)
    policy_head = layers.BatchNormalization(axis=-1)(policy_head)
    policy_head = layers.Activation('relu')(policy_head)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(MOVES, activation='softmax', name='policy')(policy_head)
    
    value_head = layers.Conv2D(1, 1)(x)
    value_head = layers.BatchNormalization(axis=-1)(value_head)
    value_head = layers.Activation('relu')(value_head)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(128, activation='relu')(value_head)
    value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)
    
    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
    return model
    
def get_model_week2_bis(convol_size=128):
    input = keras.Input(shape=(19, 19, PLANES), name='board')
    x = layers.Conv2D(32, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(input)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    # x = layers.Dropout(0.2)(x)
    
    y = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.BatchNormalization(axis=-1)(y)
    x = layers.Add()([x,y])
    x = layers.Activation('relu')(x)
    # x = layers.Dropout(0.2)(x)

    y = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.BatchNormalization(axis=-1)(y)
    x = layers.Add()([x,y])
    x = layers.Activation('relu')(x)
    # x = layers.Dropout(0.2)(x)
    
    policy_head = layers.Conv2D(2, 1)(x)
    policy_head = layers.BatchNormalization(axis=-1)(policy_head)
    policy_head = layers.Activation('relu')(policy_head)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(MOVES, activation='softmax', name='policy')(policy_head)
    
    value_head = layers.Conv2D(1, 1)(x)
    value_head = layers.BatchNormalization(axis=-1)(value_head)
    value_head = layers.Activation('relu')(value_head)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(256,
                              kernel_regularizer=regularizers.l2(0.01),
                              activation='relu')(value_head)
    value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)
    
    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
    return model
    
def get_model_week2_three(convol_size=128):
    input = keras.Input(shape=(19, 19, PLANES), name='board')
    x = layers.Conv2D(32, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(input)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    y = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.BatchNormalization(axis=-1)(y)
    x = layers.Add()([x,y])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    y = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.BatchNormalization(axis=-1)(y)
    x = layers.Add()([x,y])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    policy_head = layers.Conv2D(2, 1)(x)
    policy_head = layers.BatchNormalization(axis=-1)(policy_head)
    policy_head = layers.Activation('relu')(policy_head)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(MOVES, activation='softmax', name='policy')(policy_head)
    
    value_head = layers.Conv2D(1, 1)(x)
    value_head = layers.BatchNormalization(axis=-1)(value_head)
    value_head = layers.Activation('relu')(value_head)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(256,
                              kernel_regularizer=regularizers.l2(0.01),
                              activation='relu')(value_head)
    value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)
    
    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
    return model

def get_model_week2_four(convol_size=64, nb_resblock=8):
    input = keras.Input(shape=(19, 19, PLANES), name='board')
    x = layers.Conv2D(32, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(input)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(convol_size, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    for i in range(nb_resblock):
        y = layers.Conv2D(convol_size, 3, padding='same',
                          kernel_regularizer=regularizers.l2(0.01))(x)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Activation('relu')(y)
        y = layers.Conv2D(convol_size, 3, padding='same',
                          kernel_regularizer=regularizers.l2(0.01))(y)
        y = layers.BatchNormalization(axis=-1)(y)
        x = layers.Add()([x,y])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
    
    policy_head = layers.Conv2D(2, 1)(x)
    policy_head = layers.BatchNormalization(axis=-1)(policy_head)
    policy_head = layers.Activation('relu')(policy_head)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(MOVES, activation='softmax', name='policy')(policy_head)
    
    value_head = layers.Conv2D(1, 1)(x)
    value_head = layers.BatchNormalization(axis=-1)(value_head)
    value_head = layers.Activation('relu')(value_head)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(256,
                              kernel_regularizer=regularizers.l2(0.01),
                              activation='relu')(value_head)
    value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)
    
    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
    return model