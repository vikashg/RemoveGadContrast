import tensorflow as tf
import numpy as np 
from scipy import misc
import time
import logging 
import CostFunction as cf
import sys
import os 

# All Inputs 
data_dir = sys.argv[1]
batch_size = sys.argv[2]
num_epochs = sys.argv[3]
out_dir = sys.argv[4]

preGad_file_list_name = data_dir + 'preGad_file_list_shuffle_train.txt'
postGad_file_list_name = data_dir + 'postGad_file_list_shuffle_train.txt'
preGad_train_list = open(preGad_file_list_name, 'r').readlines()
postGad_train_list = open(postGad_file_list_name, 'r').readlines()

preGad_file_list_name = data_dir + 'preGad_file_list_shuffle_test.txt'
postGad_file_list_name = data_dir + 'postGad_file_list_shuffle_test.txt'
preGad_test_list = open(preGad_file_list_name, 'r').readlines()
postGad_test_list = open(postGad_file_list_name, 'r').readlines()

num_train_batches = int(len(preGad_train_list)/batch_size)
num_test_batches = int(len(preGad_test_list)/batch_size)

imageShape = np.array([240, 240])
numVox = np.prod(imageShape)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

log_file_name = out_dir + 'CNN_log_file.log'
logging.basicConfig(filename = log_file_name, level = logging.INFO)


x = tf.placeholder(tf.float32, [None, numVox], name = 'PostGadImage')
y = tf.placeholder(tf.float32, [None, numVox], name = 'PreGadImage')

decoded = model_lib.make_NN_model_double(x, imageShape, conv_size)

