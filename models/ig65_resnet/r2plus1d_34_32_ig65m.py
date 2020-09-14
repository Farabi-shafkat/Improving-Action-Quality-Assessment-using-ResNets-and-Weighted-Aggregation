import torch
import torch.nn as nn
import random
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)

class custom(nn.Module): 
    def __init__(self,model):
        super(custom, self).__init__()
        
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(in_features=512,out_features=256,bias=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.features(x)
        x = x.view(1, -1)
        x = self.relu(self.fc(x))
        return x

def build_model(scratch = False):
    pretrained = None
    if scratch == False:
        pretrained = True
    else:
        pretrained = False
    model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_kinetics", num_classes=400, pretrained=pretrained)
    custom_model = custom(model)
    return custom_model