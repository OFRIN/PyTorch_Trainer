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

parser.add_argument('--max_epoch', default=100, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)
parser.add_argument('--scheduler', default='step', type=str)

parser.add_argument('--print_ratio', default=0.1, type=float)
parser.add_argument('--val_ratio', default=0.5, type=float)

parser.add_argument('--tag', default='', type=str)

parser.add_argument('--loss_fn', default='ce', type=str)
parser.add_argument('--augment_fn', default='base', type=str)

parser.add_argument('--amp', default=False, type=str2bool)
parser.add_argument('--freeze', default=False, type=str2bool)
parser.add_argument('--pretrained_model_path', default='', type=str)

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
    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    if args.batch_size > 256:
        args.lr = args.batch_size / 256 * args.lr

    # 1. Transform
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    if args.augment_fn == 'base':
        train_transforms = [
            transforms.RandomResizedCrop(args.image_size),
            RandomHorizontalFlip(),
        ]

    elif args.augment_fn == 'colorjitter':
        train_transforms = [
            transforms.RandomResizedCrop(args.image_size),
            RandomHorizontalFlip(),
            train_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
        ]

    elif args.augment_fn == 'randaugment':
        train_transforms = [
            transforms.RandomResizedCrop(args.image_size),
            RandomHorizontalFlip(),
            train_transforms.append(RandAugmentMC(n=2, m=10))
        ]
    
    train_transform = transforms.Compose(train_transforms + \
        [
            Normalize(imagenet_mean, imagenet_std),
            Transpose()
        ]
    )
    test_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.CUBIC),
        transforms.CenterCrop(args.image_size),

        Normalize(imagenet_mean, imagenet_std),
        Transpose()
    ])
    
    # 2. Dataset
    class_names = read_txt(f'./data/{args.dataset_name}.txt')
    num_classes = len(class_names)

    train_dataset = Dataset_For_Folder(args.root_dir, 'train', class_names, train_transform)
    valid_dataset = Dataset_For_Folder(args.root_dir, 'validation', class_names, test_transform)

    # 3. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    
    log_iteration = int(len(train_loader) * args.print_ratio)
    val_iteration = int(len(train_loader) * args.val_ratio)
    max_iteration = args.max_epoch * len(train_loader)
    
    log_func('[i] val_iteration is {}'.format(val_iteration))
    log_func('[i] log_iteration is {}'.format(log_iteration))
    log_func('[i] max_iteration is {}'.format(max_iteration))
    
    log_func('[i] The size of training set is {}'.format(len(train_dataset)))
    log_func('[i] num_classes is {}'.format(num_classes))

    ###################################################################################
    # Network
    ###################################################################################
    model = Tagging(args.architecture, num_classes, freeze=args.freeze)
    param_groups = model.get_parameter_groups(print_fn=None)
    
    model = model.cuda()

    if not args.freeze:
        model.train()
    else:
        model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()
    
    # load pretrained model
    if args.pretrained_model_path != '':
        pretrained_model = Tagging(args.architecture, 16849)
        load_model(pretrained_model, args.pretrained_model_path)
        
        transfer_model(pretrained_model, model, 'classifier')
        
        log_func('[i] Transfer Learning ({})'.format(args.pretrained_model_path))
        del pretrained_model

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    if 'ce' in args.loss_fn:
        class_loss_fn = nn.CrossEntropyLoss().cuda()
    
    log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

    # 2. Optimizer
    if args.scheduler == 'poly':
        optimizer = PolyOptimizer([
            {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
            {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
            {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
        ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
        scheduler = None

    elif args.scheduler == 'step':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=args.nesterov)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(max_iteration * 0.5), int(max_iteration * 0.75)], gamma=0.1)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : []
    }

    train_timer = Timer()
    train_meter = Average_Meter([
        'loss', 
    ])

    eval_timer = Timer()

    def evaluate():
        accuracy_list = [[] for _ in range(num_classes)]
        
        with torch.no_grad():
            length = len(valid_loader)
            for step, (images, labels) in enumerate(valid_loader):
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

        mean_accuracy = np.mean([np.mean(accuracy_list[i]) for i in range(num_classes)]) * 100
        return mean_accuracy
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    best_valid_mean_accuracy = -1

    if args.amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)   

    for iteration in range(max_iteration):
        #################################################################################################
        images, labels = train_iterator.get()
        
        images = images.cuda()
        labels = labels.cuda()

        #################################################################################################
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(images)
            loss = class_loss_fn(logits, labels)
        #################################################################################################

        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        train_meter.add({'loss':loss.item()})
        
        #################################################################################################
        # Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)

            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            model.eval()

            mean_accuracy = evaluate()
            
            if best_valid_mean_accuracy == -1 or best_valid_mean_accuracy < mean_accuracy:
                best_valid_mean_accuracy = mean_accuracy

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'mean_accuracy' : mean_accuracy,
                'best_valid_mean_accuracy' : best_valid_mean_accuracy,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                mean_accuracy={mean_accuracy:.2f}%, \
                best_valid_mean_accuracy={best_valid_mean_accuracy:.2f}%, \
                time={time:.0f}sec'.format(**data)
            )
            
            writer.add_scalar('Evaluation/mean_accuracy', mean_accuracy, iteration)
            writer.add_scalar('Evaluation/best_valid_mean_accuracy', best_valid_mean_accuracy, iteration)

            if not args.freeze:
                model.train()

    print(args.tag)

