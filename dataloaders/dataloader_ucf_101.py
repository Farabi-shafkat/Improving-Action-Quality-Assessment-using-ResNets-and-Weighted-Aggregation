import json
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import pickle as pkl
from mc5opts import *
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

class VideoDataset(Dataset):
    def __init__(self):
        super(VideoDataset, self).__init__()
        self.video_list = glob.glob("/content/UCF-101-frames/*/*")
    
    def __getitem__(self, ix):
        transform = transforms.Compose([transforms.RandomCrop(H),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image_list = sorted(glob.glob(os.path.join(self.video_list[ix],'*.jpg')))
        images = torch.zeros(16, C, H, W)
        hori_flip = random.randint(0,1)
        start = random.randint(0,len(image_list)-17)
        for i in np.arange(0,16):
            images[i] = load_image_train(image_list[start+i], hori_flip, transform)
        data = {}
        data['video']=images
        return data 