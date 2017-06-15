import tensorflow as tf
import CNN_TF_wrapper as wrap
import numpy as np 


n_features_conv1 = 16
n_features_conv2 = 8

def make_NN_model_single(x_post, imageSize, conv_size):
    X = tf.reshape(x_post, [-1, imageSize[0], imageSize[1], 1]) #postGad
    #Convolution layers
    filter_size = tuple([conv_size]*2)
    conv1 = wrap.Convolution2d(X, (imageSize[0], imageSize[1]), 1, n_features_conv1, filter_size, activation = 'relu', name_conv = 'Conv_Layer_1')
    conv1_out = conv1.output()
    conv1_transpose = wrap.Conv2DTranspose(conv1_out, (imageSize[0], imageSize[1]), n_features_conv1, 1, filter_size , activation = 'relu', name_conv = 'Deconv_Layer_1')
    decoded = conv1_transpose.output()
    return decoded


def make_NN_model_double(x_post, imageSize, conv_size):
    X = tf.reshape(x_post, [-1, imageSize[0], imageSize[1], 1]) #postGad
    #Convolution layers
    filter_size = tuple([conv_size]*2)
    conv1 = wrap.Convolution2d(X, imageSize.tolist(), 1, n_features_conv1, filter_size, activation = 'relu', name_conv = 'Conv_Layer_1')
    conv1_out = conv1.output()
    conv2 = wrap.Convolution2d(conv1_out, imageSize.tolist(),  n_features_conv1, n_features_conv2, filter_size, activation = 'relu', name_conv ='Conv_Layer_2')
    conv2_out = conv2.output()
    conv2_transpose = wrap.Conv2DTranspose(conv2_out, imageSize.tolist(), n_features_conv2, n_features_conv1, filter_size , activation = 'relu', name_conv = 'Deconv_Layer_2')
    conv2t_out = conv2_transpose.output()
    conv1_transpose = wrap.Conv2DTranspose(conv2t_out, imageSize.tolist(), n_features_conv1, 1, filter_size , activation = 'relu', name_conv = 'Deconv_Layer_1')
    decoded = conv1_transpose.output()
    return decoded
