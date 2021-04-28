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
parser.add_argument('--train_dataset', default='OGQ_3M', type=str)

parser.add_argument('--train_data_dirs', default='../OGQ-3M_SH/', type=str)
parser.add_argument('--train_domains', default='all', type=str) # train, validation, test, *

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    # 2. Dataset
    data_dic = read_json(f'./data/{args.train_dataset}.json', encoding='utf-8')

    args.train_domains = args.train_domains.replace('all', '*')

    train_datasets = []
    for data_dir, domain in zip(args.train_data_dirs.split(','), args.train_domains.split(',')):
        if 'OPIV6' in data_dir:
            dataset_class_fn = Dataset_For_OPIV6
        else:
            dataset_class_fn = Dataset_For_OGQ_3M

        train_datasets.append(dataset_class_fn(data_dir, domain, data_dic, None))

    train_dataset = Merging_Dataset(train_datasets)

    print('[i] The size of training set is {}'.format(len(train_dataset)))