# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import DataLoader

from PIL import Image

from core import networks, datasets, losses

from tools.ai import torch_utils, training_utils, evaluating_utils
from tools.ai import augment_utils, randaugment

from tools.general import io_utils, log_utils, json_utils

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
    parser.add('batch_size', 64, int)

    parser.add('amp', False, bool)
    parser.add('tag', '', str)

    # 4. evaluation
    parser.add('task', 'multi-labels', str) # single-label or multi-labels
    parser.add('test_augment', 'resize->crop', str) 

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

    test_transforms = []
    for name in args.train_augment.split('->'):
        if name == 'resize':
            transform = transforms.Resize(args.image_size, Image.BICUBIC)
        elif name == 'crop':
            transform = transforms.CenterCrop(args.image_size)
        test_transforms.append(transform)
    
    essential_transform = [
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]
    test_transform = transforms.Compose(test_transforms + essential_transform)
    
    ###################################################################################
    # 5. Make datasets
    ###################################################################################
    data_dict = json_utils.read_json(f'./data/{args.dataset_name}.json')

    test_dataset = datasets.Dataset_For_Json(data_dict, 'test', args.task, test_transform)

    ###################################################################################
    # 6. Make loaders
    ###################################################################################
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    
    log_func('[i] The number of evaluating iterations is {}'.format(len(test_loader)))
    
    ###################################################################################
    # 7. Make Network
    ###################################################################################
    model = networks.Classifier(args.architecture, data_dict['num_classes'], pretrained=False)
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
    
    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
    log_func()
    
    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    use_multi_gpu = the_number_of_gpu > 1

    if use_multi_gpu:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    if os.path.isfile(model_path):
        torch_utils.load_model(model, model_path, parallel=use_multi_gpu)
        log_func('[i] fine-tuning from ({})'.format(model_path))
    
    #################################################################################################
    # 8. Evaluation
    #################################################################################################
    evaluator_params = {
        'model' : model,
        'loader' : test_loader,
        'class_names' : data_dict['class_names'],
        'task' : args.task,
        'amp' : args.amp
    }
    evaluator = evaluating_utils.Evaluator(**evaluator_params)
    
    # TODO: 
    if args.task == 'multi-labels':
        metric_name = 'C-F1'
    else:
        metric_name = 'mean_accuracy'
        log_func('')
    
    valid_score, eval_time = evaluator.step(detail=True)
    log_func('[i] epoch={epoch:,}, score={score:.2f}%, best_score={best_score:.2f}%, time={time:.0f}sec'.format(**data))
    
    print(args.tag)

