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
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--dataset_name', default='leak', type=str)
parser.add_argument('--root_dir', default='D:/Leak_Detection/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='efficientnet-b5', type=str)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--batch_size', default=256, type=int)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--amp', default=False, type=str2bool)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    log_dir = create_directory(f'./experiments/logs/')
    data_dir = create_directory(f'./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')
    
    log_path = log_dir + f'{args.tag}.txt'
    data_path = data_dir + f'{args.tag}.json'
    model_path = model_dir + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # 1. Transform
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    test_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.CUBIC),
        transforms.CenterCrop(args.image_size),

        Normalize(imagenet_mean, imagenet_std),
        Transpose()
    ])
    
    # 2. Dataset
    class_names = read_txt(f'./data/{args.dataset_name}.txt')
    num_classes = len(class_names)

    test_dataset = Dataset_For_Folder(args.root_dir, 'test', class_names, test_transform)

    # 3. DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    
    ###################################################################################
    # Network
    ###################################################################################
    model = Tagging(args.architecture, num_classes, pretrained=False)
    param_groups = model.get_parameter_groups(print_fn=None)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()
    
    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    #################################################################################################
    # Testing
    #################################################################################################
    eval_timer = Timer()

    def evaluate():
        accuracy_list = [[] for _ in range(num_classes)]
        
        with torch.no_grad():
            length = len(test_loader)
            for step, (images, labels) in enumerate(test_loader):
                images = images.cuda()
                labels = labels.cuda()

                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(images)
                    _, preds = torch.max(logits, 1)

                preds = get_numpy_from_tensor(preds)
                labels = get_numpy_from_tensor(labels)
                
                for i in range(images.size()[0]):
                    accuracy_list[labels[i]].append(preds[i] == labels[i])

                sys.stdout.write('\r# Evaluation [{}/{}]'.format(step + 1, length))
                sys.stdout.flush()
        print()

        mean_accuracy_list = []
        for class_name, accuracy in zip(class_names, accuracy_list):
            accuracy = np.mean(accuracy)
            mean_accuracy_list.append(accuracy)

            print('# {} = {:.2f}%'.format(class_name, accuracy * 100))
        print('# Mean Accuracy = {:.2f}%'.format(np.mean(mean_accuracy_list) * 100))
    
    mean_accuracy = evaluate()
    