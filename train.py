import numpy as np
import matplotlib.pyplot as plt
import json

import torch
from sklearn.metrics import r2_score
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision import transforms
# from torchsummary import summary

import os
import sys
#get utils

sys.path.append(os.getcwd()+'/scripts')
from Utils import *

###Networks
#single step
class RNN_Model(nn.Module):
    def __init__(self, config):
        super(RNN_Model, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.num_layers = config['num_layers']
        #default dropout = 0.1
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, dropout = config['dropout'], batch_first = True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out)
        return out
    
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, nonlinearity= 'relu')
        m.bias.data.fill_(0.0)

###multiple time steps
#one to many RNN architechture
class Base_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0.1):
        super(Base_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, dropout = dropout, batch_first = True)
        self.linear = nn.Linear(self.hidden_size, self.input_size)
        
    def forward(self, x, in_hidden = None):
        if in_hidden == None:
            out, out_hidden = self.rnn(x)
        else:
            out, out_hidden = self.rnn(x, in_hidden)
        out = self.linear(out)
        return out, out_hidden

class RNN_Model_Multiple(nn.Module):
    def __init__(self, config):
        super(RNN_Model_Multiple, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.num_layers = config['num_layers']
        self.ntimesteps = config['ntimesteps']
        self.base_rnn = Base_RNN(self.input_size, self.hidden_size, self.num_layers, config['dropout'])
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.input_size, self.output_size)
        
    def forward(self, x):
        out = x
        hidden = None
        
        #to store outputs
        outputs = []
        
        for t in range(self.ntimesteps):                
            #pass to rnn
            out, hidden = self.base_rnn(out, hidden)
            #pass to output layer
            outputs.append(self.linear(self.relu(out)))
            #skip connection
            out += x

        return torch.concat(outputs, 1)

#one to many RNN architechture
class Base_RNN_v2(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout = 0.1):
        super(Base_RNN_v2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(self.output_size, self.hidden_size, self.num_layers, dropout = dropout, batch_first = True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, in_hidden = None):
        if in_hidden == None:
            out, out_hidden = self.rnn(x)
        else:
            out, out_hidden = self.rnn(x, in_hidden)
        out = self.linear(out)
        return out, out_hidden

class RNN_Model_Multiple_v2(nn.Module):
    def __init__(self, config):
        super(RNN_Model_Multiple_v2, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.num_layers = config['num_layers']
        self.ntimesteps = config['ntimesteps']
        self.base_rnn = Base_RNN_v2(self.output_size, self.hidden_size, self.num_layers, config['dropout'])
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        #pass to first layer and activation
        x = self.relu(self.linear(x))

        #to store outputs
        outputs = []

        #set inputs to first RNN layer
        out = x
        hidden = None

        #loop through rnn layers
        for t in range(self.ntimesteps):                
            #pass to rnn
            out, hidden = self.base_rnn(out, hidden)
            #pass to output layer
            outputs.append(out)
            #skip connection
            out += x

        return torch.concat(outputs, 1)


# Config
config = dict(
    ## data
    batch_size = 128,
    dropout = 0.1,
    timesteps = [1, 5, 15, 100, 200],
    nsample = 150,
    split = [0.7, 0.2, 0.1],
    mean_std_path = "./Model1.json",#"./MeanStd/Model1.json",
    in_h2_o2 = True,
    in_conc = True,
    in_pres = False,
    out_h2_o2 = False,
    inputs_path = "input_98660.npy",
    outputs_dir = "./Out_files_npy",
    ## model
    input_size = 9,
    hidden_size = 100,
    num_layers = 2,
    output_size = 7,
    ntimesteps = 5,
    ## training
    epochs = 100,
    optimizer = 'Adam',
    SGD_momentum = 0.95,
    beta1 = 0.95, 
    beta2 = 0.9,
    lr = 0.0005,
    dict_path = "./Model1.pth",#./Models/Model1.pth",
    scheduling = False,
    mb_coeff = 0,
    model = 'v2',
    weight_decay = 0,
    return_1d_input = False
    )

def train(config = config):
    #hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #data
    #prepare dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(config)
    #obtaining mean and std of training set
    if config['mean_std_path']:
      train_x_mean, train_x_std, train_y_mean, train_y_std = load_mean_std(config['mean_std_path'], train_loader)
      mean_std = [train_x_mean.to(device), train_x_std.to(device), train_y_mean.to(device), train_y_std.to(device)]
    else:
      mean_std = None
    #model
    #initialize model
    if config['model'] == 'v1':
        model = RNN_Model_Multiple(config)
    elif config['model'] == 'v2':
        model = RNN_Model_Multiple_v2(config)

    #training
    model = model.to(device)
    criterion  = nn.MSELoss()
    e, val_loss, train_losses, val_losses = training(model, train_loader, val_loader, config, criterion, mean_std, device)
    # loss_plot(train_losses, val_losses)
    print('least val loss:', val_loss)

if __name__ == '__main__':
    print('configuration:', config)
    train()
    print('Train losses:' train_losses, 
        'Valid losses:' val_losses)