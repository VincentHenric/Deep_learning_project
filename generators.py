#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 00:41:44 2019

@author: henric
"""

import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
import golois

class GoloisSequence(tf.keras.utils.Sequence):
    def __init__(self, N=100000, batch_size=128, change_batch=5, planes=8, moves=361):
      self.batch_size = batch_size
      self.N = N
      self.change_batch = change_batch
      self.k = 0
      
      # initialize data arrays
      self.input_data = np.random.randint(2, size=(N, 19, 19, planes)).astype('float32')
      self.policy = keras.utils.to_categorical(np.random.randint(moves, size=(N,)))
      self.value = np.random.randint(2, size=(N,)).astype('float32')
      self.end = np.random.randint(2, size=(N, 19, 19, 2)).astype('float32')

    def __len__(self):
        return math.ceil(self.N / self.batch_size)

    def __getitem__(self, idx):
        # update data at the start of each change_batch epoch
        if (idx == 0) & (self.k % self.change_batch == 0):
            print("Load new batch of data...")
            golois.getBatch (self.input_data, self.policy, self.value, self.end)
            print("New batch is loaded")
        
        indices = range(idx * self.batch_size,(idx + 1) *self.batch_size)
        
        if (idx == self.__len__()-1):
            indices = range(idx * self.batch_size, self.N)
            # one epoch finished: update counter
            self.k+= 1

        return (self.input_data[indices],
                {'policy': self.policy[indices],
                 'value': self.value[indices]}
                )
