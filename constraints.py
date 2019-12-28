#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:07:52 2019

@author: henric
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint

class Symmetry(Constraint):
    """MaxNorm weight constraint.
    Constrains the weights kernel to be symmetric (typically for conv2d).
    # Arguments
        max_value: the maximum norm for the incoming weights.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
        https://www.tensorflow.org/api_docs/python/tf/reverse?version=stable
        https://www.tensorflow.org/api_docs/python/tf/reverse?version=stable
    """

    def __init__(self, kind='all'):
        self.kind = kind

    def __call__(self, w):
        if self.kind=='transpose':
            symmetrized = (K.permute_dimensions(w, (1,0,2,3)) + w) / 2
        elif self.kind=='all':
            sym_i = tf.reverse(w, [0])
            sym_j = tf.reverse(w, [1])
            sym_diag_1 = K.permute_dimensions(w, (1,0,2,3))
            sym_diag_2 = K.permute_dimensions(tf.reverse(sym_i, [1]), (1,0,2,3))
            symmetrized = (sym_i + sym_j + sym_diag_1 + sym_diag_2)/4
        else:
            raise(ValueError)
        return symmetrized

    def get_config(self):
        return {'kind': self.kind}
    
def transfer_weights(filename_in, filename_out, model_function):
    model = model_function()
    model.load_weights('models/{}.h5'.format(filename_in))
    model.compile(optimizer='adam', metrics=['acc'],
                   loss={'value': 'mse', 'policy': 'categorical_crossentropy'})
    model.save('models/{}.h5'.format(filename_out))