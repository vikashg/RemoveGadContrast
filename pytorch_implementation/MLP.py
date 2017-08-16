import numpy as np 
import sys, os, logging
import startup_header_code as startup
import torch
from torch.autograd import variable 
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


class MLP_autoencoder(nn.Module):
    
    def __init__(self, input_dimension):
        super(MLP_autoencoder, self).__init__()
        
        # Input the layer definition
        self.inp_dimen = input_dimension 
        self.h1_enc_size = 128
        self.h2_enc_size = 64
        self.h3_enc_size = 32
        
        self.h3_dec_size = 32
        self.h2_dec_size = 64
        self.h1_dec_size = 128 
        self.out_dim = input_dimension

        self.layer1 = nn.Linear(self.inp_dimen,   self.h1_enc_size, bias = True)
        self.layer2 = nn.Linear(self.h1_enc_size, self.h2_enc_size, bias = True)
        self.layer3 = nn.Linear(self.h2_enc_size, self.h3_enc_size, bias = True)

        self.layer4 = nn.Linear(self.h3_enc_size, self.h3_dec_size, bias = True)
        self.layer5 = nn.Linear(self.h3_dec_size, self.h2_dec_size, bias = True)
        self.layer6 = nn.Linear(self.h2_dec_size, self.h1_dec_size, bias = True)
        self.out_layer = nn.Linear(self.h1_dec_size, self.out_dim)

        def Encoder(self, x):
            h1_out = F.reul(self.layer1(x))
            h2_out  = F.relu(self.layer2(h1_out))
            h3_out = F.relu(self.layer3(h2_out))
            return h3_out        
    
        def Decoder(self, encoded):
            h4_out = F.relu(self.layer4(encoded))
            h5_out = F.relu(self.layer5(h4_out))
            h6_out = F.relu(self.layer6(h5_out))
            output = F.relu(self.out_layer(h6_out))
            return output
                
        

params = startup.setup_data()
postGad_train_data = params['postGad_train_data']
preGad_train_data = params['preGad_train_data']
postGad_test_data = params['postGad_test_data']
preGad_test_data = params['preGad_test_data']
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

learning_rate = step_size
momentum=0.9

data_dim = postGad_train_data.shape[1]

mlp_net = MLP_autoencoder(data_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp_net.parameters(), lr = learning_rate, momentum = momentum)


#Divide data in batches 
train_data = torch.utils.data.TensorDataset(postGad_train_data, preGad_train_data)
test_data = torch.utils.data.TensorDataset(postGad_test_data, preGad_test_data)

trainloader = torch.utils.data.DataLoader(postGad_train_data, preGad_train_data, batch_size = batch_size, shuffle = True, drop_last = False)
testloader = torch.utils.data.DataLoader(postGad_test_data, preGad_test_data, batch_size = batch_size, shuffle = True, drop_last = False)


## Training the network 

for epoch in range(num_epochs):
    training_loss = 0.0
    startup = time.time()

    for i, data in enumerate(trainloader, 0):   
        # unpack the features and the labels 
        # train_labels 
        # train_data

        features_in, features_out = Variable(features_in), Variables(features_out)
        optimizer.zero_grad()
        
        # feed forward the batch to network and compute the loss 
        output = mlp_net(features)
        loss = criterion(output, features_out)
        
        # Compute the gradient and update weights 
        loss.backward()
        optimizer.step()
    print ('[epoch: %d] loss: %.3f, elapsed time: %.2f' (epoch+1, training_loss / (i+1), time.time() - starttime))


