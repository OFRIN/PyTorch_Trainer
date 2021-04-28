# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

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
from tools.general.pickle_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

from tools.general.pickle_utils import dump_pickle

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--test_dataset', default='OGQ_3M', type=str)
parser.add_argument('--test_data_dir', default='../OPIV6_SH/', type=str)
parser.add_argument('--test_domain', default='validation', type=str) 

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='efficientnet-b0', type=str)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--image_size', default=224, type=int)

parser.add_argument('--tag', default='ResNeSt-50@Google@Focal', type=str)

parser.add_argument('--amp', default=False, type=str2bool)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    log_dir = create_directory(f'./experiments/logs/')
    model_dir = create_directory('./experiments/models/')
    pickle_dir = create_directory('./experiments/pickles/')
    
    log_path = log_dir + f'{args.tag}_for_evaluation.txt'
    model_path = model_dir + f'{args.tag}.pth'
    pickle_path = pickle_dir + f'{args.tag}.pkl'

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)
    
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
    data_dic = read_json(f'./data/{args.test_dataset}.json', encoding='utf-8')
    test_dataset = Dataset_For_OPIV6(args.test_data_dir, args.test_domain, data_dic, test_transform)

    # 3. DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    class_names = np.asarray(data_dic['class_names'])
    num_classes = data_dic['classes']
    
    ###################################################################################
    # Network
    ###################################################################################
    model = Tagging(args.architecture, num_classes)
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

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    load_model_fn()
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    eval_timer.tik()

    thresholds = list(np.arange(0.01, 1.00, 0.01))
    meter_dic = {th : {'P':np.zeros(num_classes, dtype=np.float32), 'T':np.zeros(num_classes, dtype=np.float32), 'TP':np.zeros(num_classes, dtype=np.float32)} for th in thresholds}
    
    if not os.path.isfile(pickle_path):
        with torch.no_grad():
            length = len(test_loader)
            for step, (images, labels) in enumerate(test_loader):
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

                # if step % 100 == 0:
                #     dump_pickle(pickle_path, meter_dic)

                sys.stdout.write('\r# Evaluation [{}/{}]'.format(step + 1, length))
                sys.stdout.flush()

        print()

        dump_pickle(pickle_path, meter_dic)
    else:
        meter_dic = load_pickle(pickle_path)        

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

        per_class_precision = np.mean(TP / (P + 1e-5)) * 100
        per_class_recall = np.mean(TP / (T + 1e-5)) * 100
        per_class_f1_score = 2 * ((per_class_precision * per_class_recall) / (per_class_precision + per_class_recall + 1e-5))

        overall_precision = np.sum(TP) / (np.sum(P) + 1e-5) * 100
        overall_recall = np.sum(TP) / (np.sum(T) + 1e-5) * 100
        overall_f1_score = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-5))

        op_list.append(overall_precision)
        or_list.append(overall_recall)
        o_f1_list.append(overall_f1_score)

        log_func('Th : {:.2f}, C-P : {:.2f}%, C-R : {:.2f}%, C-F1 : {:.2f}%, O-P : {:.2f}%, O-R : {:.2f}%, O-F1 : {:.2f}'.format(th * 100, per_class_precision, per_class_recall, per_class_f1_score, overall_precision, overall_recall, overall_f1_score))

    # thresholds = sorted(meter_dic.keys())[:-1]
    # o_f1_list = o_f1_list[:-1]

    best_index = np.argmax(o_f1_list)
    best_threshold = thresholds[best_index]

    best_op = op_list[best_index]
    best_or = or_list[best_index]
    best_of = o_f1_list[best_index]
    
    log_func('[Best] Th : {:.2f}, O-P : {:.2f}%, O-R : {:.2f}%, O-F1 : {:.2f}'.format(best_threshold * 100, best_op, best_or, best_of))
    log_func()

    # plt.clf()
    # plt.plot(op_list, or_list, color='orange')
    # plt.ylabel('Overall Precision')
    # plt.xlabel('Overall Recall')
    # plt.title('# Overall Precision and Recall Curve')
    # plt.savefig(fname='./experiments/' + f'{args.tag}_{args.source}_PRCurve.png')

    # plt.clf()
    # plt.plot(thresholds, o_f1_list, color='orange')
    # plt.ylabel('Overall F1-Score')
    # plt.xlabel('Threshold')
    # plt.title('# Best Threshold = {:.2f}'.format(best_threshold))
    # plt.ylim(0, 100)
    # plt.savefig(fname='./experiments/' + f'{args.tag}_{args.source}_Threshold.png') 

    # plt.legend()
    # plt.show()

