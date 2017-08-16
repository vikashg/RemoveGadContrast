import numpy as np 
import tensorflow as tf
import sys 
import os 


def setup_data(): 
    base_dir = sys.argv[1]
    out_dir = sys.argv[2]
    model_name = sys.argv[3]
    model_dir = sys.argv[4] 

    postGad_valid_file_name = base_dir + 'temp_train_postGad.npy'
    preGad_valid_file_name = base_dir + 'temp_train_preGad.npy'
    postGad_valid_data = np.load(postGad_valid_file_name)
    preGad_valid_data = np.load(preGad_valid_file_name)
    
    
    # Compute index list 
    params = {'postGad_valid_data': postGad_valid_data[0:10,:], \
              'preGad_valid_data' : preGad_valid_data[0:10,:],  \
              'out_dir' : out_dir, \
              'model_name' : model_name, \
              'model_dir': model_dir \
              }
    return params
