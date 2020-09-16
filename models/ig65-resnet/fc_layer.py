import torch
import torch.nn as nn
import numpy as np
import random
from opts import randomseed
import numpy as np
torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)

class fc_layer(nn.Module):
    def __init__(self):
        super(fc_layer, self).__init__()
        self.fc = nn.Linear(768,256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        return x