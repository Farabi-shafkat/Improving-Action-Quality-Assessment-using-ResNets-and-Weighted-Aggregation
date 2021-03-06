import torch
import torch.nn as nn
import numpy as np
import random
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)


class attention_scores(nn.Module):
    def __init__(self):
        super(attention_scores, self).__init__()
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,64)
        self.fc4 = nn.Linear(64,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
    def forward(self,x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
