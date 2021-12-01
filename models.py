import numpy as np
import pandas as pd
import json
import pickle

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data


'''
For SARSA-style learning
'''
class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        #self.batch_initial = nn.BatchNorm1d(state_size+action_size)
        self.batch1 = nn.BatchNorm1d((state_size+action_size)*2)
        
        self.linear1 = nn.Linear(state_size+action_size, (state_size+action_size)*2)
        self.linear2 = nn.Linear((state_size+action_size)*2, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, inp):
        
        #inp = self.batch_initial(inp)
        
        layer1_output = self.relu(self.batch1(self.linear1(inp)))
        output = self.linear2(layer1_output)
        return output