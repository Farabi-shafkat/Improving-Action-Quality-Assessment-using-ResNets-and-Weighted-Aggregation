import torch
import os
from MC5_training_opts import *

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

    
       
def create_ceckpoint(epoch,models,optimizer):

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict_cnn': models['cnn'].state_dict(),
        'state_dict_fc6': models['fc6'].state_dict(),
        'state_dict_reg': models['reg'].state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if with_caption:
        checkpoint['state_dict_capt'] = models['capt'].state_dict()
    if with_dive_classification:
        checkpoint['state_dict_dive'] = models['dive'].state_dict()  

    path = os.path.join(saving_dir,'checkpoint.pth')
    torch.save(checkpoint,path)

def load_ceckpoint(models,optimizer):

    path = os.path.join(saving_dir,'checkpoint.pth')
    checkpoint = torch.load(path)
    models['cnn'].load_state_dict(checkpoint['state_dict_cnn'])
    models['fc6'].load_state_dict(checkpoint['state_dict_fc6'])
    models['reg'].load_state_dict(checkpoint['state_dict_reg'])
    if with_caption:
        models['capt'].load_state_dict(checkpoint['state_dict_capt'])
    if with_dive_classification:
        models['dive'].load_state_dict(checkpoint['state_dict_dive']))  


    optimizer.load_state_dict(checkpoint['optimizer'])   
    epoch  = checkpoint['epoch']

    return (epoch,models,optimizer)
    