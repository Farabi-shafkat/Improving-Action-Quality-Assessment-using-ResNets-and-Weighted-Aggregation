feature_extractor = 'resnet2+1dig' #options are: 'resnet2+1d' 'resnet3d' 'resnet2+1dig' 'resnet50_2p1d_32'
depth = 34
clip_size = 32
with_weight = True
#save models
train_action = None
saving_dir = '...'
graph_save_dir = saving_dir

pretrained_weights_dir = '...'
# declaring random seed
randomseed = 0

# directory containing dataset annotation files; this anno_n_splits_dir make the full path
dataset_dir =  '...' 
dataset_name = 'MTLAQA' # MTLAQA or AQA7
# directory tp store train/test split lists and annotations
anno_n_splits_dir = dataset_dir + '/Ready_2_Use/MTL-AQA_split_0_data'

# directory containing extracted frames
dataset_frames_dir = '....'

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
