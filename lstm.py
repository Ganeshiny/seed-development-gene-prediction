'''
two stacked forward LSTM layers with 512 units each

Train for 5 epochs, ADAM optimizer

a learning rate 0.001 and a batch size of 128.

determine the hyper parameters

train the LSTM using GCN parameters (?) This was done by deepFri for efficiency 

'''
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import time
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class lstm_lm(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=11, num_layers=2, rnn_type='LSTM'):
        super(lstm_lm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batchsize = batch_size
        self.num_layers = num_layers
        #Initial hidden linear layer
        self.init_linear = nn.Linear(self.input_dim, self.input_dim)
        #LSTM layer
        self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        #Output layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)


    def init_hidden(self):
        ##

    def foward(self, input):
        ##

'''
Data loader for the LSTM model
'''
