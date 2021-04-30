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

parser.add_argument('--dataset_name', default='leak', type=str)
parser.add_argument('--root_dir', default='D:/Leak_Detection/', type=str)

args = parser.parse_args()

def add_txt(path):
    with open(args.root_dir + 'error.txt', 'a') as f:
        f.write(path + '\n')

class_names = read_txt(f'./data/{args.dataset_name}.txt')
num_classes = len(class_names)

# for domain in ['test', 'validation', 'train']:
for domain in ['train']:
    dataset = Dataset_For_Checking(args.root_dir, domain, class_names)

    removing_count = 0

    for image_index, (image, label) in enumerate(dataset):
        sys.stdout.write(f'\r[{image_index + 1}/{len(dataset)}] = removing {removing_count}')
        sys.stdout.flush()

        if isinstance(image, str):
            # print(image)
            try:
                os.remove(image)
            except:
                add_txt(image)

            removing_count += 1
    
    print()

