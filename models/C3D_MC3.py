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


class C3D_MC3(nn.Module):
    """
    The C3D network as described in [1].
    """
    def __init__(self):
        super(C3D_MC3, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv2d(1024, 2048, kernel_size=(3, 3), padding=(1, 1))
        self.conv3b = nn.Conv2d(2048, 1024, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4a = nn.Conv2d(1024, 2048, kernel_size=(3, 3), padding=(1, 1))
        self.conv4b = nn.Conv2d(2048, 1024, kernel_size=(3, 3), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5a = nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=(1, 1))
        self.conv5b = nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=(1, 1))
        
        self.pool5 = nn.MaxPool2d(kernel_size=( 2, 2), stride=(2, 2), padding=(1, 1))

        self.relu = nn.ReLU()


    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h= h.view(h.shape[0],-1,h.shape[-2],h.shape[-1])

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        
        

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

       # print(h.shape)
      
       # print(h.shape)


        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
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
    c3d = C3D_MC3()
    dummy_clip =  torch.zeros(1,3, 16, 112, 112)
    
    print(dummy_clip.shape)
    h = c3d(dummy_clip)
    param = c3d.parameters()
    num =0
    for p in param:
        dims = p.shape
    
        mul = 1
        for d in dims:
         
            mul *= d
        num+=mul

        
    print(num)