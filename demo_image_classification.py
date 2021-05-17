# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms

from core import networks, datasets, losses

from tools.ai import torch_utils, augment_utils, demo_utils
from tools.general import io_utils, json_utils, time_utils

if __name__ == '__main__':
    ###################################################################################
    # 1. Arguments
    ###################################################################################
    parser = io_utils.Parser()

    # 1. dataset
    parser.add('seed', 0, int)
    parser.add('num_workers', 4, int)

    parser.add('dataset_name', 'leak', str)
    
    # 2. networks
    parser.add('architecture', 'efficientnet-b5', str)

    # 3. hyperparameters
    parser.add('image_size', 456, int)

    parser.add('amp', False, bool)
    parser.add('tag', '', str)

    parser.add('task', 'multi-labels', str) # single-label or multi-labels

    parser.add('webcam_index', -1, int)
    parser.add('video_path', None, str)
    parser.add('image_dir', None, str)
    parser.add('image_path', None, str)

    parser.add('test_augment', '', str) # resize-crop

    parser.add('threshold', 0.5, float) # resize-crop
    parser.add('visualization', 'cam', str)  # cam or black-and-white

    parser.add('upper', False, bool)  # cam or black-and-white

    args = parser.get_args()

    ###################################################################################
    # 2. Make directories and pathes.
    ###################################################################################
    model_dir = io_utils.create_directory('./experiments/models/')
    model_path = model_dir + f'{args.tag}.pth'

    ###################################################################################
    # 3. Set the seed number and define log function. 
    ###################################################################################
    torch_utils.set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    ###################################################################################
    # 4. Make transformation
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    test_augment_dict = {
        'resize':transforms.Resize(args.image_size, Image.BICUBIC),
        'crop':transforms.CenterCrop(args.image_size)
    }

    test_transforms = []

    if args.test_augment != '':
        for name in args.test_augment.split('-'):
            if name in test_augment_dict.keys():
                transform = test_augment_dict[name]
            else:
                raise ValueError('unrecognize name of transform ({})'.format(name))
            
            test_transforms.append(transform)
    
        test_transform = transforms.Compose(test_transforms)
    else:
        test_transform = None

    essential_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    ###################################################################################
    # 5. Make datasets
    ###################################################################################
    data_dict = json_utils.read_json(f'./data/{args.dataset_name}.json', encoding='utf-8')

    ###################################################################################
    # 6. Make Network
    ###################################################################################
    model = networks.Classifier(args.architecture, data_dict['num_classes'])
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
    
    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
    log_func()
    
    torch_utils.load_model(model, model_path, parallel=False)
    
    #################################################################################################
    # 7. Testing
    #################################################################################################
    eval_timer = time_utils.Timer()

    def process_for_image(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return None

        if test_transform is not None:
            image = test_transform(image)
        inputs = essential_transform(image).unsqueeze(0).cuda()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits, cams = model.forward_with_cam(inputs)

                if args.task == 'multi-labels':
                    confs = F.sigmoid(logits)
                else:
                    confs = F.softmax(logits, dim=1)

                conf = confs[0].cpu().detach().numpy()
                cam = cams[0].cpu().detach().numpy()

        cond = np.sum(conf >= args.threshold) > 0

        if args.upper:
            if not cond:
                return
        else:
            if cond:
                return 
        
        image = np.asarray(image)[..., ::-1]
        h, w, c = image.shape

        cam = (cam * 255).astype(np.uint8)

        for class_name, class_activation_map, confidence in zip(data_dict['class_names'], cam, conf):
            class_activation_map = cv2.resize(class_activation_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
            if args.visualization == 'cam':
                class_activation_map = cv2.applyColorMap(class_activation_map, cv2.COLORMAP_JET)
                class_activation_map = cv2.addWeighted(image, 0.3, class_activation_map, 0.7, 0.0)
            else:
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
    elif args.image_path is not None:
        process_for_image(args.image_path)
