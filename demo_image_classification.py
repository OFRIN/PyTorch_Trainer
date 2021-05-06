# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import glob
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

from tools.general.io_utils import *
from tools.general.txt_utils import *
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
parser.add_argument('--dataset_name', default='leak', type=str)

parser.add_argument('--webcam_index', default=-1, type=int)
parser.add_argument('--video_path', default=None, type=str)
parser.add_argument('--image_dir', default=None, type=str)
parser.add_argument('--image_path', default=None, type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='efficientnet-b5', type=str)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--image_size', default=224, type=int)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--amp', default=False, type=str2bool)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + f'{args.tag}.pth'

    log_func = lambda string='': print(string)
    
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # 1. Transform
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    image_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.CUBIC),
        transforms.CenterCrop(args.image_size),

    ])

    torch_transform = transforms.Compose([
        Normalize(imagenet_mean, imagenet_std),
        Transpose(),
        torch.from_numpy
    ])
    
    # 2. Dataset
    class_names = read_txt(f'./data/{args.dataset_name}.txt')
    num_classes = len(class_names)

    ###################################################################################
    # Network
    ###################################################################################
    model = Tagging(args.architecture, num_classes, pretrained=False)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()
    
    load_model(model, model_path, parallel=False)
    
    #################################################################################################
    # Testing
    #################################################################################################
    eval_timer = Timer()

    def process_for_image(image_path):
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = image_transform(image)
        except:
            return None

        inputs = torch_transform(image).unsqueeze(0).cuda()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits, cams = model.forward_with_cam(inputs)

                conf = F.softmax(logits, dim=1)[0].cpu().detach().numpy()
                cam = cams[0].cpu().detach().numpy()
        
        image = np.asarray(image)[..., ::-1]
        h, w, c = image.shape

        cam = (cam * 255).astype(np.uint8)

        for class_name, class_activation_map, confidence in zip(class_names, cam, conf):
            class_activation_map = cv2.resize(class_activation_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # class_activation_map = cv2.applyColorMap(class_activation_map, cv2.COLORMAP_JET)
            # class_activation_map = cv2.addWeighted(image, 0.3, class_activation_map, 0.7, 0.0)

            class_activation_map = image.astype(np.float32) * (class_activation_map[..., np.newaxis] / 255.).astype(np.float32)
            class_activation_map = class_activation_map.astype(np.uint8)

            cv2.imshow(class_name, class_activation_map)

            print('# {} = {:.2f}%'.format(class_name, confidence * 100))

        cv2.imshow('image', image)
        cv2.waitKey(0)

        print()

    if args.webcam_index != -1 or args.video_path is not None:
        if args.webcam_index != -1:
            video = cv2.VideoCapture(args.webcam_index)
        elif args.video_path is not None:
            video = cv2.VideoCapture(args.video_path)

    elif args.image_dir is not None:
        for image_path in glob.glob(args.image_dir + '*'):
            process_for_image(image_path)

    else:
        process_for_image(args.image_path)
