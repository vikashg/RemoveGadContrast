#A CNN wrapper for tensorflow 

from __future__ import print_function
from __future__ import division 
from __future__ import absolute_import 

import numpy as np 
import tensorflow as tf

class Convolution2d(object):
    def __init__(self, input, input_size, in_ch, out_ch, patch_size, activation='relu'):
        self.input = input
        self.rows = input_size[0]
        self.cols = input_size[1]
        self.in_ch = in_ch
        self.activation = activation
        wshape = [patch_size[0], patch_size[1], in_ch, out_ch] 
        w_cv = tf.Variable(tf.truncated_normal(wshape, stddev = 0.1), trainable='True')
        b_cv = tf.Variable(tf.constant(0.1, shape=[out_ch]), trainable= 'True')
        self.w = w_cv
        self.b = b_cv

        self.params = [self.w, self.b]

    def output(self):
        shape4d = [-1, self.rows, self.cols, self.in_ch ] 
        x_image = tf.reshape(self.input, shape4d)
        linout = tf.nn.conv2d(x_image, self.w, strides = [1,1,1,1], padding = 'SAME')  + self.b
        
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.nn.sigmoid(linout)
        else:
            self.output = linout 

        return self.output
   
 
class Conv2DTranspose(object):
    def __init__(self, input, output_size, in_ch, out_ch, patch_size, activation='relu'):
        self.input = input
        self.rows  = output_size[0]
        self.cols  = output_size[1]
        self.out_ch = out_ch
        self.activation = activation 

        wshape = [patch_size[0], patch_size[1], out_ch , in_ch]
        w_cvt = tf.Variable(tf.truncated_normal(wshape, stddev = 0.1), trainable = True)
        b_cvt = tf.Variable(tf.constant(0.1, shape = [out_ch]), trainable = True)
        
        self.batsize = tf.shape(input)[0]
        self.w = w_cvt
        self.b = b_cvt
        self.params = [self.w, self.b]

    def output(self):
        shape4d = [self.batsize, self.rows, self.cols, self.out_ch ] 
        linout  = tf.nn.conv2d_transpose(self.input, self.w, output_shape = shape4d, strides =[ 1, 1, 1, 1], padding = 'SAME') + self.b        
        
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.nn.sigmoid(linout)
        else:
            self.output = linout 

        return self.output
