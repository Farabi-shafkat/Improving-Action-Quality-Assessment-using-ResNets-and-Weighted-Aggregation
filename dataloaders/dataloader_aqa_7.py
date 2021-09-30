

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from opts_resnet import *
from scipy.io import loadmat

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

class VideoDataset_aqa7(Dataset):

    def __init__(self, mode,action_type):
        super(VideoDataset_aqa7, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            all_annotations = loadmat('/content/content/AQA-7/Split_4/split_4_train_list.mat').get('consolidated_train_list')
            self.annotations = []
           # print(action_type)
            for i in range(len(all_annotations)):
                if all_annotations[i][0]==action_type:
                    self.annotations.append(all_annotations[i])
            self.annotations = np.asarray(self.annotations)

        else:
            all_annotations = loadmat('/content/content/AQA-7/Split_4/split_4_test_list.mat').get('consolidated_test_list')
        
            self.annotations = []
            for i in range(len(all_annotations)):
                if all_annotations[i][0]==action_type:
                    self.annotations.append(all_annotations[i])
            self.annotations = np.asarray(self.annotations)



    def __getitem__(self, ix):
        action = int(self.annotations[ix][0])
        sample_no = int(self.annotations[ix][1])
        transform = transforms.Compose([transforms.CenterCrop(H),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image_list = sorted((glob.glob(os.path.join(dataset_dir,'Actions',num2action.get(action), 'frames',
                                                    str('{:03d}'.format(sample_no)), '*.jpg'))))
        #print(sorted(glob.glob(os.path.join(dataset_dir,'Actions',num2action.get(action), 'frames',
     #                                               str('{:03d}'.format(sample_no)), '*.jpg')))  ) 
        #print(image_list)
        temporal_aug_shift = 0
        if self.mode == 'train':
            temporal_aug_shift = random.randint(temporal_aug_min, temporal_aug_max)+3
        image_list = image_list[temporal_aug_shift:temporal_aug_shift+96]

        images = torch.zeros(sample_length, C, H, W)
       # print(images.shape)
        hori_flip = 0
        for i in np.arange(0, len(image_list)):
            if self.mode == 'train':
                hori_flip += random.randint(0,1)
            
            #    print("list index ",i)
                images[i] = load_image_train(image_list[i], hori_flip, transform)
            else:
                images[i] = load_image(image_list[i], transform)

        label_final_score = self.annotations[ix][2] 
        # split_1: train_stats

        label_final_score = label_final_score/final_score_std

        data = {}
        data['video'] = images
        data['label_final_score'] = label_final_score
        data['action'] = action

        return data


    def __len__(self):
        print('No. of samples: ', len(self.annotations))
        return len(self.annotations)