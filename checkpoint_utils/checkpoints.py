import torch
torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

class checkpoints():
    def __init__(self,checkpoint_dir,saving_dir):
        self.cehckpoint_dir = checkpoint_dir
        self.saving_dir = saving_dir
    

    def create_ceckpoint(self,epoch,model,potimizer):

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)

