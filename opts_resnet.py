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

feature_extractor = 'resnet3d' #options are: 'resnet2+1d' 'resnet3d' 'resnet2+1dig' 'resnet50_2p1d_32'
depth = 34
clip_size = 16
with_weight = True
#save models
saving_dir = '/content/drive/My Drive/thesis/3d_resnet_34'
graph_save_dir = saving_dir
#pre_trained_weight_dir = '/content/drive/My Drive/Trained_Models/C3D-AVG-MTL'
pretrained_weights_dir = '/content/drive/My Drive/thesis/kenshohara_weights_my_copy'
# declaring random seed
randomseed = 0

# directory containing dataset annotation files; this anno_n_splits_dir make the full path
dataset_dir = '/content/dataset'
dataset_name = 'MTLAQA' # MTLAQA or AQA7
# directory tp store train/test split lists and annotations
anno_n_splits_dir = dataset_dir + '/Ready_2_Use/MTL-AQA_split_0_data'

# directory containing extracted frames
dataset_frames_dir = '/content/content'

# sample length in terms of no of frames
sample_length = 101

# input data dims; C3D-AVG:112; MSCADC: 180
C, H, W = 3,112,112 #3,180,180#3,112,112#
# image resizing dims; C3D-AVG: 171,128; MSCADC: 640,360
input_resize = 171,128 #640,360#171,128#
# temporal augmentation range
temporal_aug_min = -3; temporal_aug_max = 3

# score std
final_score_std = 17



max_epochs = 100

train_batch_size = 2
test_batch_size = 5

model_ckpt_interval = 1 # in epochs

base_learning_rate = 0.0001

temporal_stride = 18

test_only = True

initial_epoch = 26 # first epoch that should run 
"""
num2action = {1: diving_dir,
              2: gymvault_dir,
              3: ski_dir,
              4: snowb_dir,
              5: sync3m_dir,
              6: sync10m_dir}"""