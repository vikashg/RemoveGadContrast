import numpy as np
from scipy import misc
import time
import logging
import sys
import os

def readFileNames(data_dir, flag = 1):
    if (flag == 1):
        preGad_file_list_name = data_dir + 'preGad_file_list_shuffle_train.txt'
        postGad_file_list_name = data_dir + 'postGad_file_list_shuffle_train.txt'
        preGad_train_list = open(preGad_file_list_name, 'r').readlines()
        postGad_train_list = open(postGad_file_list_name, 'r').readlines()

        preGad_file_list_name = data_dir + 'preGad_file_list_shuffle_test.txt'
        postGad_file_list_name = data_dir + 'postGad_file_list_shuffle_test.txt'
        preGad_test_list = open(preGad_file_list_name, 'r').readlines()
        postGad_test_list = open(postGad_file_list_name, 'r').readlines()
        
        preGad_file_list_name = data_dir + 'preGad_file_list_shuffle_valid.txt'
        postGad_file_list_name = data_dir + 'postGad_file_list_shuffle_valid.txt'
        preGad_valid_list = open(preGad_file_list_name, 'r').readlines()
        postGad_valid_list = open(postGad_file_list_name, 'r').readlines()
        
        return preGad_train_list, preGad_test_list, preGad_valid_list, postGad_train_list, postGad_test_list, postGad_valid_list
    if (flag == 2):
        preGad_file_list_name = data_dir + 'preGad_file_list.txt'
        postGad_file_list_name = data_dir + 'postGad_file_list.txt'

        preGad_file_list = open(preGad_file_list_name, 'r').readlines()
        postGad_file_list = open(postGad_file_list_name, 'r').readlines()

        return preGad_file_list, postGad_file_list
