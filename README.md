This is the PyTorch code for the following paper: 

[Improving-Action-Quality-Assessment-using-ResNets-and-Weighted-Aggregation](https://arxiv.org/abs/2102.10555)

The code can be used to reproduce the resutls on MTL-AQA dataset.

Pre-trained models: The resnet's used in our code requires pre-trianed models. These mdels can be found in the following repositoreis:

[3D Resnet 34,50,101; (2+1)D-Resnet 34,50,101, all 16 frame input processing and pretrianed on Kinetics Action recognition dataset](https://github.com/kenshohara/3D-ResNets-PyTorch)

[(2+1)D Resnets 34 processing 8 and 32 frames, pretrained on kinetics and fine tuned on ig-65m dataset](https://github.com/moabitcoin/ig65m-pytorch)

## Requiremetns

* [PyTorch](https://pytorch.org/)
* Python 3
## Dataset
Download the dataset from [here](https://github.com/ParitoshParmar/MTL-AQA). Extract the necessary frames from the videos. The code expects the frames to be in a folder structure such as this:
```
~/frames
   /video0_freames
     /frame0.jpg
     .
     .
     .
     /frame1000.jpg
   /video1_frames
     /frame0.jpg
     .
     .
   .
   .
   .
  
```

## Running the code
Execute the following command:
```
python train_test_resnet.py --initial_epoch 0
```
Change the value of *initial_epoch* to resume the execution from some other epoch. 
