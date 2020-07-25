# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
# Code used in the following, also if you find it useful, please consider citing the following:
#
# @inproceedings{parmar2019and,
#   title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
#   author={Parmar, Paritosh and Tran Morris, Brendan},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={304--313},
#   year={2019}
# }

import torch
import torch.nn as nn
import numpy as np
#from opts import randomseed
import random

randomseed = 0
torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)


class STConv3d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0):
        super(STConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size),stride=(1,stride,stride),padding=(0,padding,padding))
        self.conv2 = nn.Conv3d(out_planes,out_planes,kernel_size=(kernel_size,1,1),stride=(stride,1,1),padding=(padding,0,0))

        self.bn=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn2=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu2=nn.ReLU(inplace=True)
        
        #nn.init.normal(self.conv2.weight,mean=0,std=0.01)
        #nn.init.constant(self.conv2.bias,0)
    
    def forward(self,x):
        x=self.conv(x)
        #x=self.conv2(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        return x






class C3D_SP(nn.Module):
    """
    The C3D network as described in [1].
    """
        

    def __init__(self):
        super(C3D_SP, self).__init__()

        self.conv1 = STConv3d(3, 64, kernel_size=3,stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = STConv3d(64, 128, kernel_size=3,stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = STConv3d(128, 256, kernel_size=3, padding=1,stride=1)
        self.conv3b = STConv3d(256, 256, kernel_size=3, padding=1,stride=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = STConv3d(256, 512, kernel_size=3, padding=1,stride=1)
        self.conv4b = STConv3d(512, 512, kernel_size=3, padding=1,stride=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = STConv3d(512, 512, kernel_size=3, padding=1,stride=1)
        self.conv5b = STConv3d(512, 512, kernel_size=3, padding=1,stride=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.relu = nn.ReLU()


    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.pool2(h)

        h = self.conv3a(h)
        h = self.conv3b(h)
        h = self.pool3(h)

        h = self.conv4a(h)
        h = self.conv4b(h)
        h = self.pool4(h)

        h = self.conv5a(h)
        h = self.conv5b(h)
        h = self.pool5(h)

        h = h.view(-1, 8192)
        return h

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""

if __name__ == "__main__":
    #from torchviz import make_dot
    model = C3D_SP()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    
    x = torch.zeros(1, 3, 16, 112, 112, dtype=torch.float)
    out = model(x)
    print(out.shape)

   # make_dot(out).render("attached", format="png")