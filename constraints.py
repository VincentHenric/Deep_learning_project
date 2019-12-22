#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:07:52 2019

@author: henric
"""

from tensorflow.keras import backend as K
from tensorflow.keras import Constraint

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
    """

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        symmetrized = (K.transpose(w) + w) / 2
        return symmetrized

    def get_config(self):
        return {'axis': self.axis}