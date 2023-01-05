# Look Both Ways: Self Supervising Driver Gaze Estimation and Road Scene Saliency
[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730128.pdf) | [Dataset](https://drive.google.com/drive/folders/1dANOjW_VXinhumYpddSsBTroYPxMc9Ut?usp=sharing) | [Video](https://youtu.be/GGlABGOYtFA)

ECCV 2022 (Oral Presentation)
[Presentation Video] (https://youtu.be/UCOPIDu4Jig)

Authors\
Isaac Kasahara, Simon Stent, Hyun Soo Park

![alt text](https://github.com/Kasai2020/test_read_me/blob/main/output.gif "Dataset Sample")


**Table of Contents:**<br>
1. [Overview](#overview) - purpose of dataset and code<br>
2. [Setup](#setup) - download pretrained models and resources
3. [Pretrained Models](#pretrained) - evaluate pretrained models<br>
4. [Training](#training) - steps to train new model<br>


<img src='img/github_loop.gif'>


<a name="overview"/>

## Overview

### Video Summary

[![Summary](https://img.youtube.com/vi/GGlABGOYtFA/0.jpg)](https://youtu.be/GGlABGOYtFA)

### Abstract

We present a new on-road driving dataset, called “Look Both
Ways”, which contains synchronized video of both driver faces and the
forward road scene, along with ground truth gaze data registered from
eye tracking glasses worn by the drivers. Our dataset supports the study
of methods for non-intrusively estimating a driver’s focus of attention
while driving - an important application area in road safety. A key challenge is that this task requires accurate gaze estimation, but supervised
appearance-based gaze estimation methods often do not transfer well to
real driving datasets, and in-domain ground truth to supervise them is
difficult to gather. We therefore propose a method for self-supervision of
driver gaze, by taking advantage of the geometric consistency between
the driver’s gaze direction and the saliency of the scene as observed by
the driver. We formulate a 3D geometric learning framework to enforce
this consistency, allowing the gaze model to supervise the scene saliency
model, and vice versa. We implement a prototype of our method and test
it with our dataset, to show that compared to a supervised approach it
can yield better gaze estimation and scene saliency estimation with no
additional labels.




## Dataset
[Download](https://drive.google.com/drive/folders/1dANOjW_VXinhumYpddSsBTroYPxMc9Ut?usp=sharing)

The Look Both Ways dataset features 28 different drivers under various driving conditions and environments.  The dataset contains the inward facing image of the face, the outward facing images of the scene, the scene depth image, the 2D gaze location in the scene image, the 3D gaze location relative to the inward facing camera, the left/right gaze direction with respect to the inward facing camera, the left/right 2D eye location, and the left/right 3D eye location with respect to the inward facing camera.

The download can be downloaded all at once (~300gb) or can be downloaded by each individual driving session (~10gb each)

# Code:

Our code uses the gaze training network from:
https://github.com/hysts/pl_gaze_estimation

and the saliency network from:
https://github.com/rdroste/unisal

Credits for both can be found at the bottom of this page.

## Prerequisites
- Linux
- Python >= 3.7
- NVIDIA GPU + CUDA CuDNN

<a name="setup"/>

## Setup

- Clone this repo:
```bash
git clone https://github.com/Kasai2020/look_both_ways
```

- Install dependencies:
```bash
pip install -r requirements.txt
```

- Download our dataset (350gb file)

[Download](https://drive.google.com/drive/folders/1dANOjW_VXinhumYpddSsBTroYPxMc9Ut?usp=sharing)

- Extract and place the downloaded dataset into a folder titled 'train'

- Copy train_test_split.json and place in parent folder. Structure as so:
```
    some_parent_folder
        train_test_split.json
        train
            Subject01_1_data
            Subject01_2_data
            ...
```

Navigate to pl_gaze_estimation-main/configs/examples/eth_xgaze.yaml and change DATASET_ROOT_DIR: to the folder you placed train in (some_parent_folder).



<a name="training"/>

## Training
You can train your own models for both supervised gaze and supervised saliency with our dataset, as well as train both using our self-supervised method.

### To train supervised gaze model:

You can train the gaze network with the following command (while inside the pl_gaze_estimation_main folder): 
```
python train.py --config configs/examples/eth_xgaze.yaml --options VAL.VAL_INDICES "[9,10,11]" SCHEDULER.EPOCHS 15 SCHEDULER.MULTISTEP.MILESTONES "[10, 13, 14]" DATASET.TRANSFORM.TRAIN.HORIZONTAL_FLIP false EXPERIMENT.OUTPUT_DIR exp0001
```

The training parameters can be changed in: 
```
configs/examples/eth_xgaze.yaml
```
The trained model will be saved at checkpoints in experiments/eth-xgaze/exp0001

Update the OUTPUT_DIR argument to exp0002 etc. to train and save new models.

### To train supervised saliency model:

You can train saliency network with the following command (while inside the pl_gaze_estimation_main folder):

```
python train.py --config configs/examples/eth_xgaze.yaml --options VAL.VAL_INDICES "[0, 1, 2]" SCHEDULER.EPOCHS 15 SCHEDULER.MULTISTEP.MILESTONES "[10, 13, 14]" DATASET.TRANSFORM.TRAIN.HORIZONTAL_FLIP false EXPERIMENT.OUTPUT_DIR exp0001
```

The training parameters can be changed in: 

```
configs/examples/eth_xgaze.yaml
```
The trained model will be saved at checkpoints in experiments/eth-xgaze/exp0001

Update the OUTPUT_DIR argument to exp0002 etc. to train and save new models.

### To train using our self-supervised method:

The models from the supervised gaze and supervised saliency methods above can be improved using our self-supervised method.  Change line XX in train.py to the pretrained gaze model location, and line XX in train.py to the pretrained saliency model location.

You can now train both the saliency and gaze network with the following command (while inside the pl_gaze_estimation_main folder):

```
python train.py --config configs/examples/eth_xgaze.yaml --options VAL.VAL_INDICES "[0, 1, 2]" SCHEDULER.EPOCHS 15 SCHEDULER.MULTISTEP.MILESTONES "[10, 13, 14]" DATASET.TRANSFORM.TRAIN.HORIZONTAL_FLIP false EXPERIMENT.OUTPUT_DIR exp0001
```

<a name="pretrained"/>

## Evaluate with pretrained models
Our pretrained models can be found here:

link

```
python test.py --config configs/examples/eth_xgaze.yaml --options VAL.VAL_INDICES "[0, 1, 2]" SCHEDULER.EPOCHS 15 SCHEDULER.MULTISTEP.MILESTONES "[10, 13, 14]" DATASET.TRANSFORM.TRAIN.HORIZONTAL_FLIP false EXPERIMENT.OUTPUT_DIR exp0503
```


## Cite

```
@inproceedings{kasahara22eccv,
    author = {Kasahara, Isaac and Stent, Simon and Park, Hyun Soo},
    title = {Look Both Ways: Self-Supervising Driver Gaze Estimation and Road Scene Saliency},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2022}
}
```


