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

feature_extractor = 'resnet2+1dig' #options are: 'resnet2+1d' 'resnet3d' 'resnet2+1dig' 'resnet50_2p1d_32'
depth = 34
clip_size = 32
with_weight = True
#save models
train_action = None
saving_dir = '/content/drive/My Drive/thesis/AQA7_resent2p1dD_34_32frame_with_att/action{}'
graph_save_dir = saving_dir
#pre_trained_weight_dir = '/content/drive/My Drive/Trained_Models/C3D-AVG-MTL'
pretrained_weights_dir = '/content/drive/My Drive/thesis/kenshohara_weights_my_copy'
# declaring random seed
randomseed = 0

# directory containing dataset annotation files; this anno_n_splits_dir make the full path
dataset_dir =  '/content/content/AQA-7' # or /content/content/AQA-7 or '/content/dataset'
dataset_name = 'AQA7' # MTLAQA or AQA7
# directory tp store train/test split lists and annotations
anno_n_splits_dir = dataset_dir + '/Ready_2_Use/MTL-AQA_split_0_data'

# directory containing extracted frames
dataset_frames_dir = '/content/content'

# sample length in terms of no of frames
sample_length = 100

# input data dims; C3D-AVG:112; MSCADC: 180
C, H, W = 3,112,112 #3,180,180#3,112,112#
# image resizing dims; C3D-AVG: 171,128; MSCADC: 640,360
input_resize = 171,128 #640,360#171,128#
# temporal augmentation range
temporal_aug_min = -3; temporal_aug_max = 3

# score std
final_score_std = 17





train_batch_size = 2
test_batch_size = 5

model_ckpt_interval = 1 # in epochs

base_learning_rate = 0.0001

temporal_stride = 18

test_only = True

initial_epoch = None # first epoch that should run 
max_epoch = 50

num2action = {1: "diving",
              2: "gym_vault",
              3: "ski_big_air",
              4: "snowboard_big_air",
              5: "sync_diving_3m",
              6: "sync_diving_10m"}
