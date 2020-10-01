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

#save models
saving_dir = '/content/drive/My Drive/resnet_basedAQA_more_fc_layers_diff_lr_diff_modules_32_frames_16_step'
graph_save_dir = '/content/drive/My Drive/resnet_basedAQA_more_fc_layers_diff_lr_diff_modules_32_frames_16_step'
#pre_trained_weight_dir = '/content/drive/My Drive/Trained_Models/C3D-AVG-MTL'
# declaring random seed
randomseed = 0

# directory containing dataset annotation files; this anno_n_splits_dir make the full path
dataset_dir = '/content/dataset'

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

initial_epoch = 6 # first epoch that should run 

