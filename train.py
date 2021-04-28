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
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--train_dataset', default='OGQ_3M', type=str)

parser.add_argument('--train_data_dirs', default='../OGQ-3M_SH/', type=str)
parser.add_argument('--train_domains', default='all', type=str) # train, validation, test, *

parser.add_argument('--test_data_dir', default='../OPIV6_SH/', type=str)
parser.add_argument('--test_domain', default='validation', type=str) 

parser.add_argument('--early_stop', default=True, type=str2bool) 

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='efficientnet-b0', type=str)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--max_epoch', default=100, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--print_ratio', default=0.1, type=float)
parser.add_argument('--val_ratio', default=0.5, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)

parser.add_argument('--gamma', default=4, type=int)
parser.add_argument('--alpha', default=1, type=float)

parser.add_argument('--losses', default='focal', type=str)

parser.add_argument('--source', default='google', type=str) # for opiv6

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
    
    train_transforms = [
        transforms.RandomResizedCrop(args.image_size),
        RandomHorizontalFlip(),
    ]

    if 'colorjitter' in args.augment:
        train_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    if 'randaugment' in args.augment:
        train_transforms.append(RandAugmentMC(n=2, m=10))
    
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
    data_dic = read_json(f'./data/{args.train_dataset}.json', encoding='utf-8')

    args.train_domains = args.train_domains.replace('all', '*')

    train_datasets = []
    for data_dir, domain in zip(args.train_data_dirs.split(','), args.train_domains.split(',')):
        if 'OPIV6' in data_dir:
            dataset_class_fn = Dataset_For_OPIV6
        else:
            dataset_class_fn = Dataset_For_OGQ_3M

        train_datasets.append(dataset_class_fn(data_dir, domain, data_dic, train_transform))

    train_dataset = Merging_Dataset(train_datasets)

    if 'OGQ' in args.test_data_dir:
        valid_dataset = Dataset_For_OGQ_3M(args.test_data_dir, args.test_domain, data_dic, test_transform)
    else:
        valid_dataset = Dataset_For_OPIV6(args.test_data_dir, args.test_domain, data_dic, test_transform)

    # train_dataset = Dataset_For_PIXTA_18M(args.train_data_dir, args.train_domain, data_dic, train_transform)
    # valid_dataset = Dataset_For_PIXTA_18M(args.test_data_dir, args.test_domain, data_dic, test_transform)
    
    # 3. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    
    log_iteration = int(len(train_loader) * args.print_ratio)
    val_iteration = int(len(train_loader) * args.val_ratio)
    max_iteration = args.max_epoch * len(train_loader)
    
    log_func('[i] val_iteration is {}'.format(val_iteration))
    log_func('[i] log_iteration is {}'.format(log_iteration))
    log_func('[i] max_iteration is {}'.format(max_iteration))

    class_names = np.asarray(data_dic['class_names'])
    num_classes = data_dic['classes']
    
    log_func('[i] The size of training set is {}'.format(len(train_dataset)))
    log_func('[i] num_classes is {}'.format(num_classes))

    ###################################################################################
    # Network
    ###################################################################################
    model = Tagging(args.architecture, num_classes)
    param_groups = model.get_parameter_groups(print_fn=None)
    
    model = model.cuda()
    model.train()

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

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    # load pretrained model
    # if os.path.isfile(option['pretrained_model_path']):
    #     log_func('[i] load pretrained model path : {}'.format(option['pretrained_model_path']))

    #     pretrained_model = Classifier(args.model_name, train_dataset.classes, args.pretrained)
    #     load_model(pretrained_model, option['pretrained_model_path'])

    #     transfer_model(pretrained_model, model)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    if 'ce' in args.losses:
        class_loss_fn = F.multilabel_soft_margin_loss
    elif 'focal' in args.losses:
        class_loss_fn = Focal_Loss(gamma=args.gamma, alpha=args.alpha).cuda()
    elif 'lsep' in args.losses:
        class_loss_fn = LSEP_Loss().cuda()

    log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

    # 2. Optimizer
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)   
    
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
    thresholds = list(np.arange(0.10, 1.00, 0.10))

    def evaluate():
        meter_dic = {th : {'P':np.zeros(num_classes, dtype=np.float32), 'T':np.zeros(num_classes, dtype=np.float32), 'TP':np.zeros(num_classes, dtype=np.float32)} for th in thresholds}
        
        with torch.no_grad():
            length = len(valid_loader)
            for step, (images, labels) in enumerate(valid_loader):
                images = images.cuda()
                labels = labels.cuda()

                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(images)
                    preds = torch.sigmoid(logits)

                preds = get_numpy_from_tensor(preds)
                labels = get_numpy_from_tensor(labels)
                
                for i in range(images.size()[0]):
                    for th in thresholds:
                        pred = (preds[i] >= th).astype(np.float32)
                        gt = labels[i]

                        meter_dic[th]['P'] += pred
                        meter_dic[th]['T'] += gt
                        meter_dic[th]['TP'] += (gt * (pred == gt)).astype(np.float32)

                sys.stdout.write('\r# Evaluation [{}/{}]'.format(step + 1, length))
                sys.stdout.flush()
        print()

        op_list = []
        or_list = []
        o_f1_list = []

        for th in sorted(meter_dic.keys()):
            data = meter_dic[th]

            P = data['P']
            T = data['T']
            TP = data['TP']

            # FP = (P - TP) / (T + P - TP + 1e-10)
            # FN = (T - TP) / (T + P - TP + 1e-10)
            # TN = ALL - (T + TP + (P - TP) + (T - TP))
            # print(np.mean(TN), np.mean(T), np.mean((P - TP)))

            # TPR = np.mean(TP / (TP + FN + 1e-10))
            # FPR = np.mean(FP / (FP + TN + 1e-10))

            # print('TH : {:.2f}, TPR : {:.2f}, FPR : {:.2f}'.format(th, TPR, FPR))

            overall_precision = np.sum(TP) / (np.sum(P) + 1e-5) * 100
            overall_recall = np.sum(TP) / (np.sum(T) + 1e-5) * 100
            overall_f1_score = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-5))

            op_list.append(overall_precision)
            or_list.append(overall_recall)
            o_f1_list.append(overall_f1_score)

        best_index = np.argmax(o_f1_list)
        best_threshold = thresholds[best_index]

        best_op = op_list[best_index]
        best_or = or_list[best_index]
        best_of = o_f1_list[best_index]

        return best_threshold, best_op, best_or, best_of
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    best_valid_f1_score = -1

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

        train_meter.add({'loss':loss.item()})
        
        #################################################################################################
        # Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
        # if True:
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

            th, valid_precision, valid_recall, valid_f1_score = evaluate()
            
            if best_valid_f1_score == -1 or best_valid_f1_score < valid_f1_score:
                best_valid_f1_score = valid_f1_score

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'th' : th,
                'valid_precision' : valid_precision,
                'valid_recall' : valid_recall,
                'valid_f1_score' : valid_f1_score,
                'best_valid_f1_score' : best_valid_f1_score,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                th={th:.2f}, \
                valid_precision={valid_precision:.2f}%, \
                valid_recall={valid_recall:.2f}%, \
                valid_f1_score={valid_f1_score:.2f}%, \
                best_valid_f1_score={best_valid_f1_score:.2f}%, \
                time={time:.0f}sec'.format(**data)
            )
            
            writer.add_scalar('Evaluation/threshold', th, iteration)
            writer.add_scalar('Evaluation/valid_precision', valid_precision, iteration)
            writer.add_scalar('Evaluation/valid_recall', valid_recall, iteration)
            writer.add_scalar('Evaluation/valid_f1_score', valid_f1_score, iteration)
            writer.add_scalar('Evaluation/best_valid_f1_score', best_valid_f1_score, iteration)

            model.train()

            if args.early_stop:
                break
            
    print(args.tag)

