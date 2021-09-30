import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_MTLAQA import VideoDataset
from dataloaders.dataloader_aqa_7 import VideoDataset_aqa7
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
import argparse as arg


import opts_resnet 
from utils import utils_1
import numpy as np
from make_graph import draw_graph

from models.models.fc_layer import fc_layer
from models.models.regressor import score_regressor
from models.models.r2plus1d_34_32_ig65m import build_model
from models.models.attention_scores import attention_scores
from models.models.resnet2p1d import build_model as resnet2p1d_build_model
from models.models.resnet3d import build_model as resnet3d_build_model
from models.models.resnet50_2p1d_32 import build_model as resnet50_2p1d_32_build_model

torch.manual_seed(opts_resnet.randomseed); torch.cuda.manual_seed_all(opts_resnet.randomseed); random.seed(opts_resnet.randomseed); np.random.seed(opts_resnet.randomseed)
torch.backends.cudnn.deterministic=True
#torch.backends.cudnn.benchmark = False 


def update_graph_data(epoch,tr_loss,ts_loss,rho,action_rho):
    opts_resnet.graph_save_dir = opts_resnet.graph_save_dir.format(opts_resnet.train_action)
    print('saving grapg data at ',opts_resnet.graph_save_dir)
    path = os.path.join(opts_resnet.graph_save_dir,'graph_data_{}.npy'.format(opts_resnet.train_action))
    if epoch==0 :
        graph_data = np.array([[tr_loss],[ts_loss],[rho] ])
    else:
        graph_data = np.load(path,allow_pickle=True)
        graph_data = np.append( graph_data,[[tr_loss],[ts_loss],[rho] ],axis= 1)

    np.save(path,  graph_data)

def save_model(model, model_name, epoch, path):
    print('sabing {} at {}'.format(model_name,path))
    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)


def train_phase(train_dataloader, optimizer, criterions, epoch):
    accumulated_loss = 0
    criterion_final_score = criterions['criterion_final_score']; penalty_final_score = criterions['penalty_final_score']

    model_CNN.train()
    model_my_fc6.train()
    model_score_regressor.train()
    model_attention_scores.train()

    iteration = 0
    for data in train_dataloader:
        true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
        difficulty = 1 
        if opts_resnet.dataset_name=='MTL_AQA':
            difficulty = data['DD'].unsqueeze_(1).type(torch.FloatTensor).cuda()
        video = data['video'].transpose_(1, 2).cuda()

        batch_size, C, frames, H, W = video.shape
        clip_feats = torch.Tensor([]).cuda()
        att_scores = torch.Tensor([]).cuda()
        for i in np.arange(0, frames - opts_resnet.clip_size+1,opts_resnet.clip_size):
            clip = video[:, :, i:i + opts_resnet.clip_size, :, :]
            clip_feats_cnn = model_CNN(clip)   ## none X 512
            clip_feats_temp = model_my_fc6(clip_feats_cnn)
            
            

            if opts_resnet.with_weight:
                att_score_temp = model_attention_scores(clip_feats_temp)

                att_score_temp=att_score_temp.unsqueeze(2)
                att_scores = torch.cat((att_scores, att_score_temp), 2)
            else:
                att_score_temp = torch.ones_like(clip_feats_temp)
                att_score_temp=att_score_temp.unsqueeze(2)
                att_scores = torch.cat((att_scores, att_score_temp), 2)
            
            clip_feats_temp=clip_feats_temp.unsqueeze(2)  ## none X 512 X 1
            
            
            clip_feats = torch.cat((clip_feats, clip_feats_temp), 2) ## none X 512 X 3
            

        pred_final_score = difficulty*model_score_regressor(clip_feats,att_scores)
        

        loss = criterion_final_score(pred_final_score, true_final_score) + penalty_final_score(pred_final_score, true_final_score)
       
     
        
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
        ""

        model_CNN.eval()
        model_my_fc6.eval()
        model_score_regressor.eval()
        model_attention_scores.eval()

        iteration = 0
        for data in test_dataloader:
            true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
            
           #for act,sc in zip(data['action'].data.numpy(),data['label_final_score'].data.numpy()):
            #    action_wise_scores_true[act].append(sc)
            true_scores.extend(data['label_final_score'].data.numpy())
            difficulty = 1 
            if opts_resnet.dataset_name=='MTL_AQA':
                difficulty = data['DD'].unsqueeze_(1).type(torch.FloatTensor).cuda()
            video = data['video'].transpose_(1, 2).cuda()

            batch_size, C, frames, H, W = video.shape
            clip_feats = torch.Tensor([]).cuda()
            att_scores = torch.Tensor([]).cuda()
            for i in np.arange(0, frames - opts_resnet.clip_size+1,opts_resnet.clip_size):
                clip = video[:, :, i:i + opts_resnet.clip_size, :, :]
                clip_feats_cnn = model_CNN(clip)   ## none X 512
                clip_feats_temp = model_my_fc6(clip_feats_cnn)
                
                

                
                if opts_resnet.with_weight:
                    att_score_temp = model_attention_scores(clip_feats_temp)
                    att_score_temp=att_score_temp.unsqueeze(2)
                    att_scores = torch.cat((att_scores, att_score_temp), 2)
                else:
                    att_score_temp = torch.ones_like(clip_feats_temp)
                    att_score_temp=att_score_temp.unsqueeze(2)
                    att_scores = torch.cat((att_scores, att_score_temp), 2)
                
                
                clip_feats_temp=clip_feats_temp.unsqueeze(2)  ## none X 512 X 1
            
            
                clip_feats = torch.cat((clip_feats, clip_feats_temp), 2) ## none X 512 X 3

      
            temp_final_score = difficulty*model_score_regressor(clip_feats,att_scores)
        
            pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])
         

            loss = criterion_final_score(temp_final_score, true_final_score)+ penalty_final_score(temp_final_score, true_final_score)

            accumulated_loss += loss.item()
            iteration += 1
    
        
        rho, p = stats.spearmanr(pred_scores, true_scores)
        print('Predicted scores: ', pred_scores)
        print('True scores: ', true_scores)
        print('Correlation : ', rho)





        return (accumulated_loss/iteration,rho,rho)


def main():

    if not os.path.exists(opts_resnet.graph_save_dir):
        os.mkdir(opts_resnet.graph_save_dir)
    if not os.path.exists(opts_resnet.saving_dir):
        os.mkdir(opts_resnet.saving_dir)
    

    
    parameters_2_optimize = [
        {'params': model_CNN.parameters(),'lr':0.00001},
        {'params': model_attention_scores.parameters(), 'lr':0.00001},
        {'params': list(model_my_fc6.parameters()) + list(model_score_regressor.parameters())}
    ]
    optimizer = optim.Adam(parameters_2_optimize, lr=0.0001)



    if opts_resnet.initial_epoch>0 and os.path.exists((os.path.join(opts_resnet.saving_dir, '%s_%d.pth' % ('optimizer',opts_resnet.initial_epoch-1)))):
        optimizer_state_dic =  torch.load((os.path.join(opts_resnet.saving_dir, '%s_%d.pth' % ('optimizer', opts_resnet.initial_epoch-1))))  


    criterions = {}
    criterion_final_score = nn.MSELoss()
    penalty_final_score = nn.L1Loss()
    criterions['criterion_final_score'] = criterion_final_score
    criterions['penalty_final_score'] = penalty_final_score

    
    train_dataset = None 
    test_dataset = None 
    if opts_resnet.dataset_name == 'MTLAQA':
        train_dataset = VideoDataset('train')
        test_dataset = VideoDataset('test')
    elif opts_resnet.dataset_name == 'AQA7':
        train_dataset = VideoDataset_aqa7('train',opts_resnet.train_action)
        test_dataset = VideoDataset_aqa7('test',opts_resnet.train_action)     
    train_dataloader = DataLoader(train_dataset, batch_size=opts_resnet.train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opts_resnet.test_batch_size, shuffle=False)
    print('Length of train loader: ', len(train_dataloader))
    print('Length of test loader: ', len(test_dataloader))
    print('Training set size: ', len(train_dataloader)*opts_resnet.train_batch_size,';    Test set size: ', len(test_dataloader)*opts_resnet.test_batch_size)

    # actual training, testing loops
    for epoch in range(opts_resnet.initial_epoch,opts_resnet.max_epoch):
       # 
        print('-------------------------------------------------------------------------------------------------------')
        for param_group in optimizer.param_groups:
            print('Current learning rate: ', param_group['lr'])

        tr_loss = train_phase(train_dataloader, optimizer, criterions, epoch)
        ts_loss,rho,action_rho = test_phase(test_dataloader,criterions)
       # scheduler.step()

        if (epoch+1) % opts_resnet.model_ckpt_interval == 0: # save models every 5 epochs
            save_model(model_CNN, 'model_CNN', epoch, opts_resnet.saving_dir)
            save_model(model_my_fc6, 'model_my_fc6', epoch, opts_resnet.saving_dir)
            save_model(model_score_regressor, 'model_score_regressor', epoch, opts_resnet.saving_dir)
            save_model(optimizer,'optimizer',epoch,opts_resnet.saving_dir)
            if opts_resnet.with_weight:
                save_model(model_attention_scores,'model_attention_scores',epoch,opts_resnet.saving_dir)
           # save_model(scheduler,'scheduler',epoch,saving_dir)
        print("training loss: {} test loss: {} rho: {}".format(tr_loss,ts_loss,rho))
        update_graph_data(epoch,tr_loss,ts_loss,rho,action_rho)   
        #draw_graph()


if __name__ == '__main__':
   
    parser = arg.ArgumentParser()
    parser.add_argument('--initial_epoch',type = int)
    parser.add_argument('--train_action',type = int)
    args = vars(parser.parse_args())
    opts_resnet.initial_epoch = args['initial_epoch']
    opts_resnet.train_action = args['train_action']
    opts_resnet.saving_dir = opts_resnet.saving_dir.format(opts_resnet.train_action)
    model_CNN = None
    if opts_resnet.feature_extractor == 'resnet2+1dig':
        assert opts_resnet.depth==34 
        model_CNN = build_model(pretrained=True,ex_type = opts_resnet.feature_extractor[0:-2]+'_{}'.format(opts_resnet.clip_size))
    elif opts_resnet.feature_extractor == 'resnet2+1d':
        assert opts_resnet.clip_size==16
        model_CNN = resnet2p1d_build_model(opts_resnet.depth)
    elif opts_resnet.feature_extractor == 'resnet3d':
        assert opts_resnet.clip_size==16
        model_CNN = resnet3d_build_model(opts_resnet.depth)
    elif opts_resnet.feature_extractor == 'resnet50_2p1d_32':
        assert opts_resnet.clip_size==32 and opts_resnet.depth == 50
        model_CNN = resnet50_2p1d_32_build_model(opts_resnet.depth)
   

    if opts_resnet.initial_epoch > 0:
        model_CNN_pretrained_dict = torch.load((os.path.join(opts_resnet.saving_dir, '%s_%d.pth' % ('model_CNN', opts_resnet.initial_epoch-1))))
        model_CNN.load_state_dict(model_CNN_pretrained_dict)
    
    model_CNN = model_CNN.cuda()

 
    model_my_fc6 = fc_layer()
 
    if opts_resnet.initial_epoch > 0:
        model_fc6_pretrained_dict = torch.load((os.path.join(opts_resnet.saving_dir, '%s_%d.pth' % ('model_my_fc6', opts_resnet.initial_epoch-1))))  
        model_my_fc6.load_state_dict(model_fc6_pretrained_dict)
    model_my_fc6.cuda()


    model_score_regressor = score_regressor()
    if opts_resnet.initial_epoch > 0:
        model_score_regressor_pretrained_dict = torch.load((os.path.join(opts_resnet.saving_dir, '%s_%d.pth' % ('model_score_regressor', opts_resnet.initial_epoch-1))))
        model_score_regressor.load_state_dict(model_score_regressor_pretrained_dict)
    model_score_regressor = model_score_regressor.cuda()
    

    model_attention_scores = None
    if opts_resnet.with_weight:
        model_attention_scores = attention_scores()
        if opts_resnet.initial_epoch > 0:
            model_attention_scores_pretrained_dict = torch.load((os.path.join(opts_resnet.saving_dir, '%s_%d.pth' % ('model_attention_scores', opts_resnet.initial_epoch-1))))
            model_attention_scores.load_state_dict(model_attention_scores_pretrained_dict)

        model_attention_scores.cuda()
    print('Using Final Score Loss')
    print(opts_resnet.train_action)

    main()




