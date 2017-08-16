import numpy as np 
import tensorflow as tf
import sys 
import os 


def setup_data(): 
    base_dir = sys.argv[1]
    out_dir = sys.argv[2]
    batch_size = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
    model_name = sys.argv[5]
    model_id = sys.argv[6]
    step = float(sys.argv[7])    
    activation = sys.argv[8]
    opt_flag = sys.argv[9]
    momentum_val = float(sys.argv[10])    
    loss_flag = sys.argv[11]
    data_flag = sys.argv[12]

    if data_flag == 'Full':
        postGad_train_file_name = base_dir + 'train_postGad.npy'
        preGad_train_file_name = base_dir + 'train_preGad.npy'
        postGad_test_file_name = base_dir + 'valid_postGad.npy'
        preGad_test_file_name = base_dir + 'valid_preGad.npy'
    else:
        print('Running_reduced files')
        postGad_train_file_name = base_dir + 'temp_train_postGad.npy'
        preGad_train_file_name = base_dir + 'temp_train_preGad.npy'
        postGad_test_file_name = base_dir + 'temp_valid_postGad.npy'
        preGad_test_file_name = base_dir + 'temp_valid_preGad.npy'
        
    postGad_train_data = np.load(postGad_train_file_name)
    preGad_train_data = np.load(preGad_train_file_name)
    postGad_test_data = np.load(postGad_test_file_name)
    preGad_test_data = np.load(preGad_test_file_name)
    log_file_name = out_dir + 'log_file.txt'
    
    num_train = postGad_train_data.shape[0]
    num_batches = int(num_train/batch_size)
    
   

    temp_img = postGad_train_data[0,:]
    numVox = postGad_train_data.shape[1]
    imageShape =np.sqrt(numVox) 
    full_model_name = out_dir + model_name
    
    # Compute index list 
    index_list = np.arange(0, num_train, batch_size, dtype=int)
    params = {'postGad_train_data': postGad_train_data, \
              'preGad_train_data' : preGad_train_data,  \
              'postGad_test_data' : postGad_test_data, \
              'preGad_test_data'  : preGad_test_data, \
              'base_dir' : base_dir, \
              'num_batches': num_batches,  \
              'numVox': numVox, \
              'index_list': index_list, \
              'logfilename': log_file_name, \
              'model_name': full_model_name, \
              'out_dir': out_dir, \
              'num_epochs': num_epochs, \
              'model_id' : model_id, \
              'step_size': step, \
              'activation': activation, \
              'batch_size' : batch_size, \
              'momentum' : momentum_val, \
              'opt_flag' : opt_flag, \
              'loss_flag' : loss_flag  } 

    return params
