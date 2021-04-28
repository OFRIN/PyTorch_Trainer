# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *
from core.losses import *
from core.tag_utils import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--dataset', default='OGQ_3M', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='efficientnet-b5', type=str)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--image_size', default=456, type=int)
parser.add_argument('--tag', default='EfficientNet-b5@Focal@OGQ-3M', type=str)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + f'{args.tag}.pth'

    log_func = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    
    # 1. Transform
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    test_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.CUBIC),
        transforms.CenterCrop(args.image_size),

        Normalize(imagenet_mean, imagenet_std),
        Transpose(),

        torch.from_numpy
    ])
    
    # 2. Dataset
    data_dic = read_json(f'./data/{args.dataset}.json', encoding='utf-8')
    
    class_names = np.asarray(data_dic['class_names'])
    num_classes = data_dic['classes']
    
    log_func('[i] num_classes is {}'.format(num_classes))

    ###################################################################################
    # Network
    ###################################################################################
    model = Tagging(args.architecture, num_classes, pretrained=False)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    load_model(model, model_path)
    
    ###################################################################################
    # Video
    ###################################################################################
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        # preprocessing
        image = Image.fromarray(frame[..., ::-1])
        image = test_transform(image)

        # inference
        logits = model(image.unsqueeze(0).cuda())

        # postprocessing
        predictions = torch.sigmoid(logits)[0].detach().cpu().numpy()

        # threshold is 0.3
        tags = class_names[predictions >= 0.3]

        # visualize
        print(tags)

        cv2.imshow('show', frame)
        cv2.waitKey(1)