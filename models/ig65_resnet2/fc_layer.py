import torch
import torch.nn as nn
import numpy as np
import random
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)

class fc_layer(nn.Module):
    def __init__(self):
        super(fc_layer, self).__init__()
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,128)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x