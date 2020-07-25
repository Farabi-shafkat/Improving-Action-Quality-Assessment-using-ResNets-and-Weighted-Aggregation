import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_MSCADC import VideoDataset
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.MSCADC.body import C3D_dilated_body
from models.MSCADC.head_fs_2 import C3D_dilated_head_fs
from models.MSCADC.head_dive_classifier import C3D_dilated_head_classifier
from models.MSCADC.head_captions import S2VTModel

import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_C3DAVG import VideoDataset
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.C3DAVG.C3D_altered import C3D_altered
from models.C3DAVG.my_fc6 import my_fc6
from models.C3DAVG.score_regressor import score_regressor
from models.C3DAVG.dive_classifier import dive_classifier
from models.C3DAVG.S2VTModel import S2VTModel
from opts import *
from utils import utils_1
import numpy as np
from make_graph import draw_graph
import numpy as np
from models.C3D_MC5 import C3D_MC5
from models.c3d_seperable_batch_norm import C3D_SP
from models.C3D_MC3 import C3D_MC3
from models.C3D_MC4 import C3D_MC4



model_CNN = C3D_altered()

model_score_regressor = score_regressor()


model_fc6 = my_fc6()

params = 0

model_parameters_back = filter(lambda p: p.requires_grad, model_CNN.parameters())
params_back = sum([np.prod(p.size()) for p in model_parameters_back])


model_parameters_reg = filter(lambda p: p.requires_grad, model_score_regressor.parameters())
params_reg = sum([np.prod(p.size()) for p in model_parameters_reg])


model_parameters_fc6 = filter(lambda p: p.requires_grad, model_fc6.parameters())
params_fc6 = sum([np.prod(p.size()) for p in model_parameters_fc6])

params = params_back+ params_reg + params_fc6

print("c3d"," ",params," c3d->",params_back," fc->",params_fc6," reg->",params_reg )




model_CNN = C3D_MC5()

model_score_regressor = score_regressor()


model_fc6 = my_fc6()

params = 0

model_parameters_back = filter(lambda p: p.requires_grad, model_CNN.parameters())
params_back = sum([np.prod(p.size()) for p in model_parameters_back])


model_parameters_reg = filter(lambda p: p.requires_grad, model_score_regressor.parameters())
params_reg = sum([np.prod(p.size()) for p in model_parameters_reg])


model_parameters_fc6 = filter(lambda p: p.requires_grad, model_fc6.parameters())
params_fc6 = sum([np.prod(p.size()) for p in model_parameters_fc6])

params = params_back+ params_reg + params_fc6

print("mc5"," ",params," c3d->",params_back," fc->",params_fc6," reg->",params_reg )




model_CNN = C3D_MC4()

model_score_regressor = score_regressor()


model_fc6 = my_fc6()

params = 0

model_parameters_back = filter(lambda p: p.requires_grad, model_CNN.parameters())
params_back = sum([np.prod(p.size()) for p in model_parameters_back])


model_parameters_reg = filter(lambda p: p.requires_grad, model_score_regressor.parameters())
params_reg = sum([np.prod(p.size()) for p in model_parameters_reg])


model_parameters_fc6 = filter(lambda p: p.requires_grad, model_fc6.parameters())
params_fc6 = sum([np.prod(p.size()) for p in model_parameters_fc6])

params = params_back+ params_reg + params_fc6

print("mc4"," ",params," c3d->",params_back," fc->",params_fc6," reg->",params_reg )





model_CNN = C3D_MC3()

model_score_regressor = score_regressor()


model_fc6 = my_fc6()

params = 0

model_parameters_back = filter(lambda p: p.requires_grad, model_CNN.parameters())
params_back = sum([np.prod(p.size()) for p in model_parameters_back])


model_parameters_reg = filter(lambda p: p.requires_grad, model_score_regressor.parameters())
params_reg = sum([np.prod(p.size()) for p in model_parameters_reg])


model_parameters_fc6 = filter(lambda p: p.requires_grad, model_fc6.parameters())
params_fc6 = sum([np.prod(p.size()) for p in model_parameters_fc6])

params = params_back+ params_reg + params_fc6

print("mc3"," ",params," c3d->",params_back," fc->",params_fc6," reg->",params_reg )



model_CNN = C3D_SP()

model_score_regressor = score_regressor()


model_fc6 = my_fc6()

params = 0

model_parameters_back = filter(lambda p: p.requires_grad, model_CNN.parameters())
params_back = sum([np.prod(p.size()) for p in model_parameters_back])


model_parameters_reg = filter(lambda p: p.requires_grad, model_score_regressor.parameters())
params_reg = sum([np.prod(p.size()) for p in model_parameters_reg])


model_parameters_fc6 = filter(lambda p: p.requires_grad, model_fc6.parameters())
params_fc6 = sum([np.prod(p.size()) for p in model_parameters_fc6])

params = params_back+ params_reg + params_fc6

print("sp"," ",params," c3d->",params_back," fc->",params_fc6," reg->",params_reg )




