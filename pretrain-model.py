import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_ucf_101 import VideoDataset
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
import numpy as np

from models.C3DAVG.C3D_altered import C3D_altered
from models.C3D_MC5 import C3D_MC5
from models.c3d_seperable import C3D_SP
from models.C3D_MC3 import C3D_MC3
from models.C3D_MC4 import C3D_MC4
from utils import utils_1
from make_graph import draw_graph
from pre_training_opts import * 

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

def save_model(model, model_name, epoch, path):
    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)

def update_graph_data(epoch,tr_loss):
    path = os.path.join(graph_save_dir,'graph_data_{}.npy'.format(model_type))
    if epoch==0 :
        graph_data = np.array([[tr_loss],[0]])
    else:
        graph_data = np.load(path)
        graph_data = np.append( graph_data,np.array([[tr_loss],[0]]),axis= 1)

    np.save(path,  graph_data)

def train_phase(train_dataloader, optimizer, criterion, epoch):
    accumulated_loss = 0

    model_CNN.eval()
    model_CNN_student.train()

    iteration = 0

    for data in train_dataloader:
       
        video = data['video'].transpose_(1, 2)
        if torch.cuda.is_available():
            video = video.cuda()
        batch_size, C, frames, H, W = video.shape
        #clip_feats = torch.Tensor([]).cuda()
        with torch.no_grad():
            clip_feats_CNN = model_CNN(video)
        clip_feats_CNN_student = model_CNN_student(video)
        loss = criterion(clip_feats_CNN,clip_feats_CNN_student)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accumulated_loss += loss.item()
        if iteration % 20 == 0:
             print('Epoch: ', epoch, ' Iter: ', iteration, ' Loss: ', loss)
        iteration+=1
    
    return accumulated_loss/iteration 

def main():
    parameters_2_optimize = list(model_CNN_student.parameters())
    parameters_2_optimize_named = list(model_CNN_student.named_parameters())
   
    print('[INFO]Parameters that will be learnt: ', parameters_2_optimize_named)
    optimizer = optim.Adam(parameters_2_optimize,lr=0.0001)
    criterion = nn.MSELoss()
    dataset = VideoDataset()
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    print('[INFO] Length of dataloader: ', len(dataloader))
    print('Training set size: ', len(dataloader)*train_batch_size)

    for epoch in range(initial_epoch,100):
        print('-------------------------------------------------------------------------------------------------------')
        
        for param_group in optimizer.param_groups:
            print('Current learning rate: ', param_group['lr'])

        tr_loss=train_phase(dataloader, optimizer, criterion, epoch)
        update_graph_data(epoch,tr_loss)   
        draw_graph()
        print('[INFO]  epoch {}  loss: {}'.format(epoch,tr_loss))
        if (epoch+1) % model_ckpt_interval == 0:
            save_model(model_CNN_student, 'model_CNN_{}'.format(model_type), epoch, saving_dir)
    


if __name__=='__main__':

    model_CNN = C3D_altered()
    model_CNN_dict = model_CNN.state_dict()
    model_CNN_pretrained_dict = torch.load('/content/c3d.pickle')
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    model_CNN.load_state_dict(model_CNN_dict)


    if not os.path.exists(graph_save_dir):
        os.mkdir(graph_save_dir)
    if not os.path.exists(graph_save_dir):
        os.mkdir(saving_dir)

    if torch.cuda.is_available():
        model_CNN = model_CNN.cuda()

    model_CNN_student = None
    if model_type == 'mc5':
        model_CNN_student = C3D_MC5()
    if model_type == 'mc4':
        model_CNN_student = C3D_MC4()
    if model_type == 'mc3':
        model_CNN_student = C3D_MC3()
    if model_type == 'sp':
        model_CNN_student = C3D_SP()

    if initial_epoch!=0:
        model_CNN_student_dict = torch.load((os.path.join(saving_dir, '%s_%d.pth' % ('model_{}'.format(model_type), initial_epoch-1))))
        model_CNN_student.load_state_dict( model_CNN_student_dict)
    if torch.cuda.is_available():
        model_CNN_student.cuda()
    print('[INFO] loaded all the models')

    main()