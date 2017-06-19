import tensorflow as tf
import numpy as np 
from scipy import misc
import time
import logging 
import CostFunction as cf
import sys
import os 
import NN_model_library as model_lib

# All Inputs 
data_dir = sys.argv[1]
batch_size = int(sys.argv[2])
num_epochs = int(sys.argv[3])
out_dir = sys.argv[4]
conv_type = int(sys.argv[5])

conv_size = 5
step_size=0.001

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

num_train_batches =2
num_test_batches =2

imageShape = np.array([240, 240])
numVox = np.prod(imageShape)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

log_file_name = out_dir + 'CNN_log_file.log'
logging.basicConfig(filename = log_file_name, level = logging.INFO)
x = tf.placeholder(tf.float32, [None, numVox], name = 'PostGadImage')
y = tf.placeholder(tf.float32, [None, numVox], name = 'PreGadImage')

if (conv_type == 2):
    decoded = model_lib.make_NN_model_double(x, imageShape, conv_size)
if (conv_type == 3):
    print('Using 3 stacks')
    CONV_SIZE=np.array([10, 7, 5])
    decoded = model_lib.make_NN_model_3(x, imageShape, CONV_SIZE)

x_imag = tf.reshape(x, [-1, imageShape[0], imageShape[1], 1])
y_imag = tf.reshape(y, [-1, imageShape[0], imageShape[1], 1])

with tf.name_scope('Images') as scope:
    tf.summary.image('PostGadImage', x_imag)
    tf.summary.image('PreGadImage', y_imag)
    tf.summary.image('PredictedImage', decoded)

with tf.name_scope("CostFunction") as scope:
    loss = cf.SSD(y, decoded, numVox)
    train_loss_summary = tf.summary.scalar("Train_Loss", loss)
    test_loss_summary = tf.summary.scalar("Test_loss", loss)
with tf.name_scope("Training") as scope:
    train_op = tf.train.AdamOptimizer(step_size).minimize(loss)

merged = tf.summary.merge_all()  
init = tf.global_variables_initializer()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
    sess.run(init)
    print('Starting session')
    
    for k in range(num_epochs):
        time_start = time.time()
        total_train_loss = 0
        for j in range(num_train_batches):
            postGad_batch = cf.CreateBatch(postGad_train_list, j, batch_size)
            preGad_batch = cf.CreateBatch(preGad_train_list, j, batch_size)
            batch_dict = {x: postGad_batch, y: preGad_batch}
            _, cost, summary = sess.run([train_op, loss, merged], feed_dict = batch_dict)
            total_train_loss += cost
            print('Loss: ', cost)
            summary_writer.add_summary(summary)
            x_imag = tf.reshape(postGad_batch, [batch_size, imageShape[0], imageShape[1]])
            y_imag = tf.reshape(preGad_batch, [batch_size, imageShape[0], imageShape[1]])

        total_test_loss = 0
        for j in range(num_test_batches):
            postGad_test_batch = cf.CreateBatch(postGad_test_list, j, batch_size)
            preGad_test_batch = cf.CreateBatch(preGad_test_list, j, batch_size)
            test_batch_dict = {x:postGad_test_batch, y:preGad_test_batch}
            test_loss_summ_, test_loss, predicted_image = sess.run([test_loss_summary, loss, decoded ], feed_dict = test_batch_dict)
            total_test_loss = total_test_loss + test_loss
            summary_writer.add_summary(test_loss_summ_)
        
        step_loss_str = 'Epoch: ' + str(k) + ' of ' + str(num_epochs) + ' Training Loss: ' + str(total_train_loss) + ' Test loss: ' + str(total_test_loss)
        print(step_loss_str)
        logging.info(step_loss_str)
