import numpy as np
from scipy import misc
import time
import logging
import sys
import os

def readFileNames(data_dir):
    preGad_file_list_name = data_dir + 'preGad_file_list_shuffle_train.txt'
    postGad_file_list_name = data_dir + 'postGad_file_list_shuffle_train.txt'
    preGad_train_list = open(preGad_file_list_name, 'r').readlines()
    postGad_train_list = open(postGad_file_list_name, 'r').readlines()

    preGad_file_list_name = data_dir + 'preGad_file_list_shuffle_test.txt'
    postGad_file_list_name = data_dir + 'postGad_file_list_shuffle_test.txt'
    preGad_test_list = open(preGad_file_list_name, 'r').readlines()
    postGad_test_list = open(postGad_file_list_name, 'r').readlines()

    return preGad_train_list, preGad_test_list, postGad_train_list, postGad_test_list
