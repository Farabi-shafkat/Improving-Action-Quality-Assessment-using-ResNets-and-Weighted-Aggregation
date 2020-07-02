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
from models.C3DAVG.C3D_altered import C3D_altered
from models.C3DAVG.my_fc6 import my_fc6
from models.C3DAVG.score_regressor import score_regressor
from models.C3DAVG.dive_classifier import dive_classifier
from models.C3DAVG.S2VTModel import S2VTModel
from opts import *
from utils import utils_1
import numpy as np
from make_graph import draw_graph


torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

def update_graph_data(epoch,tr_loss,ts_loss):
    path = os.path.join(graph_save_dir,'graph_data.npy')
    if epoch==0 :
        graph_data = np.array([[tr_loss],[ts_loss]])
    else:
        graph_data = np.load(path)
        graph_data = np.append( graph_data,np.array([[tr_loss],[ts_loss]]),axis= 1)

    np.save(path,  graph_data)





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

            for i in np.arange(0, frames - 17, 16):
                clip = video[:, :, i:i + 16, :, :]
                clip_feats_temp = model_CNN(clip)
                clip_feats_temp.unsqueeze_(0)
                clip_feats_temp.transpose_(0, 1)
                clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)
            clip_feats_avg = clip_feats.mean(1)

            sample_feats_fc6 = model_my_fc6(clip_feats_avg)
            temp_final_score = model_score_regressor(sample_feats_fc6)
            pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])
          
                
            loss_final_score = (criterion_final_score(temp_final_score, true_final_score) + penalty_final_score(temp_final_score, true_final_score))
            loss = 0
            loss += loss_final_score
            accumulated_loss += loss.item()
            iteration += 1
        
        
        rho, p = stats.spearmanr(pred_scores, true_scores)
        print('Predicted scores: ', pred_scores)
        print('True scores: ', true_scores)
        print('Correlation: ', rho)
        return accumulated_loss/iteration,rho


def main():


  

    criterions = {}
    criterion_final_score = nn.MSELoss()
    penalty_final_score = nn.L1Loss()
    criterions['criterion_final_score'] = criterion_final_score
    criterions['penalty_final_score'] = penalty_final_score
    if with_dive_classification:
        criterion_dive_classifier = nn.CrossEntropyLoss()
        criterions['criterion_dive_classifier'] = criterion_dive_classifier
    if with_caption:
        criterion_caption = utils_1.LanguageModelCriterion()
        criterions['criterion_caption'] = criterion_caption

    
    test_dataset = VideoDataset('test')

    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
  
    print('Length of test loader: ', len(test_dataloader))
    print(   'Test set size:' , len(test_dataloader)*test_batch_size)
    
    path = '/content/drive/My Drive/authors_code_graphs/graph_data.npy'
    graph_data = np.load(path)
    # actual training, testing loops
    rhos = []
    for epoch in range(initial_epoch,graph_data.shape[1]):
       # 
        print('-------------------------------------------------------------------------------------------------------')
        model_fc6_pretrained_dict = torch.load((os.path.join(saving_dir, '%s_%d.pth' % ('model_my_fc6', epoch))))  
        model_my_fc6.load_state_dict(model_fc6_pretrained_dict)


        model_score_regressor_pretrained_dict = torch.load((os.path.join(saving_dir, '%s_%d.pth' % ('model_score_regressor', epoch))))
        model_score_regressor.load_state_dict(model_score_regressor_pretrained_dict)


        model_CNN_pretrained_dict = torch.load((os.path.join(saving_dir, '%s_%d.pth' % ('model_CNN', epoch))))
        model_CNN.load_state_dict(model_CNN_pretrained_dict )

        ts_loss,rho=test_phase(test_dataloader,criterions)
        
        print('epoch {} ts_loss->{} , rho:{}'.format(epoch,ts_loss,rho))

        graph_data[1][epoch]=ts_loss
        path_save = '/content/drive/My Drive/authors_code_graphs/graph_data_restored.npy'
        #np.save(path_save,  graph_data)
        rhos.append(rho)
    
    rho = np.array(rho,dtype=float)
    plt.plot(np.arrange(0,99),rho,linewidth= 3,color = 'blue',marker ='o',style='dashed',label='rho')
    plt.xlabel("Epoch #")
    plt.ylabel("sp.cprrelation")
    plt.legend()
    plt.show()
    np.save(rho,'/content/c3d_authors_implementation/rhos.npy')
         
        


if __name__ == '__main__':
    # loading the altered C3D backbone (ie C3D upto before fc-6)

    model_CNN = C3D_altered()
    model_CNN_dict = model_CNN.state_dict()
    if initial_epoch == 0:
        model_CNN_pretrained_dict = torch.load('/content/c3d.pickle')
    else:
        model_CNN_pretrained_dict = torch.load((os.path.join(saving_dir, '%s_%d.pth' % ('model_CNN', initial_epoch-1))))

    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    model_CNN.load_state_dict(model_CNN_dict)
    model_CNN = model_CNN.cuda()

    # loading our fc6 layer
    model_my_fc6 = my_fc6()
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