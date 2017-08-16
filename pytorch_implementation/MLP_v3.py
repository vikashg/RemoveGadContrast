import numpy as np 
import sys, os, logging
import startup_header_code as startup
import torch
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import time
import logging

class MLP_autoencoder(nn.Module):
    
    def __init__(self, input_dimension):
        super(MLP_autoencoder, self).__init__()
        
        # Input the layer definition
        self.inp_dimen = input_dimension 
        self.h0_enc_size = 256
        self.h1_enc_size = 128
        self.h2_enc_size = 64
        self.h3_enc_size = 32
        
        self.h3_dec_size = 32
        self.h2_dec_size = 64
        self.h1_dec_size = 128
        self.h0_dec_size = 256 
        self.out_dim = input_dimension

        self.layer1 = nn.Linear(self.inp_dimen,   self.h0_enc_size, bias = True)
        self.layer2 = nn.Linear(self.h0_enc_size, self.h1_enc_size, bias = True)
        self.layer3 = nn.Linear(self.h1_enc_size, self.h2_enc_size, bias = True)
        self.layer4 = nn.Linear(self.h2_enc_size, self.h3_enc_size, bias = True)

        self.layer4 = nn.Linear(self.h3_enc_size, self.h3_dec_size, bias = True)
        self.layer5 = nn.Linear(self.h3_dec_size, self.h2_dec_size, bias = True)
        self.layer6 = nn.Linear(self.h2_dec_size, self.h1_dec_size, bias = True)
        self.layer7 = nn.Linear(self.h1_dec_size, self.h0_dec_size, bias = True)
        self.out_layer = nn.Linear(self.h0_dec_size, self.out_dim)

    def forward(self, x):
        inp = x.view(-1, self.inp_dimen)
        h1_out = F.relu(self.layer1(inp))
        h2_out  = F.relu(self.layer2(h1_out))
        h3_out = F.relu(self.layer3(h2_out))

        h4_out = F.relu(self.layer4(h3_out))
        h5_out = F.relu(self.layer5(h4_out))
        h6_out = F.relu(self.layer6(h5_out))
        output = F.relu(self.out_layer(h6_out))
        return output
                
        


params = startup.setup_data()
postGad_train_data = params['postGad_train_data']
preGad_train_data = params['preGad_train_data']
postGad_valid_data = params['postGad_test_data']
preGad_valid_data = params['preGad_test_data']
base_dir = params['base_dir']
num_batches = params['num_batches']
num_epochs = params['num_epochs']
numVox = params['numVox']
index_list = params['index_list']
model_name = params['model_name']
logfilename = params['logfilename']
out_dir = params['out_dir']
model_id = params['model_id']
step_size = params['step_size']
activation = params['activation']
batch_size = params['batch_size']
opt_flag = params['opt_flag']
loss_flag = params['loss_flag']

num_epochs = num_epochs
batch_size = batch_size
learning_rate = step_size
momentum = params['momentum']
data_dim = numVox

#Define Logger and output files 
loss_function_file = out_dir + 'Loss.txt'
fid_loss = open(loss_function_file, 'w')
logging.basicConfig(filename = logfilename, level = logging.INFO)
 

## Define the network
mlp_net = MLP_autoencoder(data_dim)
if loss_flag == 'SSD':
    criterion = nn.MSELoss()
    print(loss_flag)
elif loss_flag == 'KL':
    criterion = nn.KLDivLoss()
    print(loss_flag)
elif loss_flag == 'BCE':
    criterion = nn.BCELoss()
    print(loss_flag)
else: 
    criterion = nn.MSELoss()
    print('Using Default Loss')
   
    
optimizer = torch.optim.SGD(mlp_net.parameters(), lr = learning_rate, momentum = momentum)


print(postGad_train_data.shape)
postGad_train_tensor = torch.from_numpy(postGad_train_data).float()
preGad_train_tensor = torch.from_numpy(preGad_train_data).float()

postGad_valid_tensor = torch.from_numpy(postGad_valid_data).float()
preGad_valid_tensor = torch.from_numpy(preGad_valid_data).float()

valid_data = torch.utils.data.TensorDataset(postGad_valid_tensor, preGad_valid_tensor)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle = True, drop_last = False)

#Divide data in batches 
train_data = torch.utils.data.TensorDataset(postGad_train_tensor, preGad_train_tensor)
trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = False)



## Training the network 
for epoch in range(num_epochs):
    training_loss = 0.0
    starttime = time.time()

    for i, data_i in enumerate(trainloader, 0):   
        postGad, preGad = data_i  # Comes from previous line 

        features_in, features_out = Variable(postGad), Variable(preGad)
        optimizer.zero_grad()
        
        # feed forward the batch to network and compute the loss 
        output = mlp_net(features_in)
        loss = criterion(output, features_out)
        
        # Compute the gradient and update weights 
        loss.backward()
        optimizer.step()
        training_loss += loss.data[0]

    ## Validation Set 
    validation_loss = 0.0
    for i, valid_data_i in enumerate(validloader, 0):
        postGad_v_data, preGad_v_data = valid_data_i
        features_v_i, features_v_o = Variable(postGad_v_data), Variable(preGad_v_data)
        output = mlp_net(features_v_i)
        loss = criterion(output, features_v_o)
        validation_loss += loss.data[0]


    print ('[epoch: %d] Training loss: %.3f, Validation loss %.3f, elapsed time: %.2f' %(epoch+1, training_loss / (i+1), validation_loss/(i+1), time.time() - starttime))
    log_str = 'Epoch: ' + str(epoch + 1) + ' Training Loss: ' + str(training_loss/(i+1)) + ' Validation loss: ' + str(validation_loss/(i+1)) + ' Time Elapsed ' + str(time.time() - starttime) 
    disp_str = str(training_loss/(i+1)) + '\t' + str(validation_loss/(i+1)) + '\n'
    fid_loss.write(disp_str)
    logging.info(log_str)

## Save Model 
full_model_path = model_name + '.pkl'
torch.save(mlp_net.state_dict(), full_model_path)
