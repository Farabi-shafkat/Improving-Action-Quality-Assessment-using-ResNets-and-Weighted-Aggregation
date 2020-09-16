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

import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_C3DAVG import VideoDataset
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
#from models.C3DAVG.C3D_altered import C3D_altered
from models.C3D_MC5 import C3D_MC5
from models.C3DAVG.my_fc6 import my_fc6
from models.C3DAVG.score_regressor import score_regressor
from models.C3DAVG.dive_classifier import dive_classifier
from models.C3DAVG.S2VTModel import S2VTModel
from opts_resnet import *
from utils import utils_1
import numpy as np
from make_graph import draw_graph
from models.C3DAVG.C3D_altered import C3D_altered
from models.C3D_MC5 import C3D_MC5
from models.c3d_seperable_batch_norm import C3D_SP
from models.C3D_MC3 import C3D_MC3
from models.C3D_MC4 import C3D_MC4
from models.ig65_resnet.fc_layer import fc_layer
from models.ig65_resnet.regressor import score_regressor
from models.ig65_resnet.r2plus1d_34_32_ig65m import build_model



torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

def update_graph_data(epoch,tr_loss,ts_loss,rho):
    path = os.path.join(graph_save_dir,'graph_data.npy')
    if epoch==0 :
        graph_data = np.array([[tr_loss],[ts_loss],[rho]])
    else:
        graph_data = np.load(path)
        graph_data = np.append( graph_data,np.array([[tr_loss],[ts_loss],[rho]]),axis= 1)

    np.save(path,  graph_data)

def save_model(model, model_name, epoch, path):
    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)


def train_phase(train_dataloader, optimizer, criterions, epoch):
    accumulated_loss = 0
    criterion_final_score = criterions['criterion_final_score']; penalty_final_score = criterions['penalty_final_score']

    model_CNN.train()
    model_my_fc6.train()
    model_score_regressor.train()
 

    iteration = 0
    for data in train_dataloader:
        true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
     
        video = data['video'].transpose_(1, 2).cuda()

        batch_size, C, frames, H, W = video.shape
        clip_feats = torch.Tensor([]).cuda()

        for i in np.arange(0, frames - 17, 32):
                clip = video[:, :, i:i + 32, :, :]
                clip_feats_temp = model_CNN(clip)   ## none X 512
                clip_feats_temp.unsqueeze_(2)  ## none X 512 X 1
                #clip_feats_temp.transpose_(0, 1)
                clip_feats = torch.cat((clip_feats, clip_feats_temp), 2) ## none X 512 X 3
        clip_feats_avg = clip_feats.mean(2)  ##none X512
        #clip_feats_avg = clip_feats.mean(1)

        sample_feats_fc6 = model_my_fc6(clip_feats_avg)

        pred_final_score = model_score_regressor(sample_feats_fc6)
        

        loss = criterion_final_score(pred_final_score, true_final_score)
                           # + penalty_final_score(pred_final_score, true_final_score)
        #loss = 0
        #loss = loss_final_score
     
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accumulated_loss += loss.item()
        if iteration % 50 == 0:
            print('Epoch: ', epoch, ' Iter: ', iteration, ' Loss: ', loss,end="")
          
            print(' ')
        iteration += 1
    return accumulated_loss/iteration


def test_phase(test_dataloader,criterions):
    print('In testphase...')
    accumulated_loss = 0
    criterion_final_score = criterions['criterion_final_score']; penalty_final_score = criterions['penalty_final_score']

    with torch.no_grad():
        pred_scores = []; true_scores = []


        model_CNN.eval()
        model_my_fc6.eval()
        model_score_regressor.eval()

        iteration = 0
        for data in test_dataloader:
            true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
            true_scores.extend(data['label_final_score'].data.numpy())
         
            video = data['video'].transpose_(1, 2).cuda()

            batch_size, C, frames, H, W = video.shape
            clip_feats = torch.Tensor([]).cuda()

            for i in np.arange(0, frames - 17, 32):
                clip = video[:, :, i:i + 32, :, :]
                clip_feats_temp = model_CNN(clip)   ## none X 512
                clip_feats_temp.unsqueeze_(2)  ## none X 512 X 1
                #clip_feats_temp.transpose_(0, 1)
                clip_feats = torch.cat((clip_feats, clip_feats_temp), 2) ## none X 512 X 3
            clip_feats_avg = clip_feats.mean(2)  ##none X512

            sample_feats_fc6 = model_my_fc6(clip_feats_avg)
            temp_final_score = model_score_regressor(sample_feats_fc6)
            pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])

                
            loss = criterion_final_score(temp_final_score, true_final_score)# + penalty_final_score(temp_final_score, true_final_score))
          #  loss = 0
          #  loss += loss_final_score
            accumulated_loss += loss.item()
            iteration += 1
    
        
        rho, p = stats.spearmanr(pred_scores, true_scores)
        print('Predicted scores: ', pred_scores)
        print('True scores: ', true_scores)
        print('Correlation: ', rho)
        return (accumulated_loss/iteration,rho)


def main():

    if not os.path.exists(graph_save_dir):
        os.mkdir(graph_save_dir)
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
        

   # parameters_2_optimize = (list(model_CNN.parameters()) + list(model_my_fc6.parameters()) +
                           #list(model_score_regressor.parameters()))
    #parameters_2_optimize_named = (list(model_CNN.named_parameters()) + list(model_my_fc6.named_parameters()) +
                                  # list(model_score_regressor.named_parameters()))

    parameters_2_optimize = [

        {'params':model_CNN.parameters(),'lr': 0},
        {'params':model_my_fc6.parameters()},
        {'params':model_score_regressor.parameters()}

    ]
    optimizer = optim.Adam(parameters_2_optimize, lr=1e-3)

    if initial_epoch>0 and os.path.exists((os.path.join(saving_dir, '%s_%d.pth' % ('optimizer', initial_epoch-1)))):
        optimizer_state_dic =  torch.load((os.path.join(saving_dir, '%s_%d.pth' % ('optimizer', initial_epoch-1))))  
        optimizer.load_state_dict(optimizer_state_dic)

  #  print('Parameters that will be learnt: ', parameters_2_optimize_named)
    #print('training model {}'.format(model_type))
   
    criterions = {}
    criterion_final_score = nn.MSELoss()
    penalty_final_score = nn.L1Loss()
    criterions['criterion_final_score'] = criterion_final_score
    criterions['penalty_final_score'] = penalty_final_score


    train_dataset = VideoDataset('train')
    test_dataset = VideoDataset('test')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    print('Length of train loader: ', len(train_dataloader))
    print('Length of test loader: ', len(test_dataloader))
    print('Training set size: ', len(train_dataloader)*train_batch_size,';    Test set size: ', len(test_dataloader)*test_batch_size)

    # actual training, testing loops
    for epoch in range(initial_epoch,100):
       # 
        print('-------------------------------------------------------------------------------------------------------')
        for param_group in optimizer.param_groups:
            print('Current learning rate: ', param_group['lr'])

        tr_loss=train_phase(train_dataloader, optimizer, criterions, epoch)
        ts_loss,rho=test_phase(test_dataloader,criterions)

        if (epoch+1) % model_ckpt_interval == 0: # save models every 5 epochs
            save_model(model_CNN, 'model_CNN', epoch, saving_dir)
            save_model(model_my_fc6, 'model_my_fc6', epoch, saving_dir)
            save_model(model_score_regressor, 'model_score_regressor', epoch, saving_dir)
            save_model(optimizer,'optimizer',epoch,saving_dir)


        print("training loss: {} test loss: {} rho: {}".format(tr_loss,ts_loss,rho))
        update_graph_data(epoch,tr_loss,ts_loss,rho)   
        #draw_graph()


if __name__ == '__main__':
    # loading the altered C3D backbone (ie C3D upto before fc-6)

   

    model_CNN = build_model(scratch = True)

    #model_CNN_dict = model_CNN.state_dict()

    if initial_epoch > 0:
        model_CNN_pretrained_dict = torch.load((os.path.join(saving_dir, '%s_%d.pth' % ('model_CNN', initial_epoch-1))))
        model_my_fc6.load_state_dict(model_fc6_pretrained_dict)
        model_CNN.load_state_dict(model_CNN_dict)
    
    model_CNN = model_CNN.cuda()

    # loading our fc6 layer
    model_my_fc6 = fc_layer()
    #model_fc6_dict = model_my_fc6.state_dict()
    if initial_epoch > 0:
        model_fc6_pretrained_dict = torch.load((os.path.join(saving_dir, '%s_%d.pth' % ('model_my_fc6', initial_epoch-1))))  
        model_my_fc6.load_state_dict(model_fc6_pretrained_dict)
    model_my_fc6.cuda()

    # loading our score regressor
    model_score_regressor = score_regressor()
    if initial_epoch > 0:
        model_score_regressor_pretrained_dict = torch.load((os.path.join(saving_dir, '%s_%d.pth' % ('model_score_regressor', initial_epoch-1))))
        model_score_regressor.load_state_dict(model_score_regressor_pretrained_dict)
    model_score_regressor = model_score_regressor.cuda()
    
    print('Using Final Score Loss')


    main()