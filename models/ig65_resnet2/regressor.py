
import torch
import torch.nn as nn
import numpy as np
import random
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)

class score_regressor(nn.Module):
    def __init__(self):
        super(score_regressor, self).__init__()
        self.fc_final_score = nn.Linear(128,1)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x ,att):
        x = x * self.softmax(att)
        x = torch.sum(x,2)
        final_score = self.fc_final_score(x)

        return final_score