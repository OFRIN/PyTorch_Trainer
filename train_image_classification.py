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

    # 4. training
    parser.add('max_epochs', 100, int)

    parser.add('lr', 0.1, float)
    parser.add('wd', 1e-4, float)
    parser.add('nesterov', True, bool)

    parser.add('optimizer', 'sgd', str)
    parser.add('scheduler', 'step', str)

    parser.add('valid_ratio', 5.0, float)

    parser.add('min_image_size', 320, int)
    parser.add('max_image_size', 640, int)

    parser.add('randaugment_n', 2, int)
    parser.add('randaugment_m', 10, int)

    parser.add('loss', 'ce', str)
    parser.add('task', 'multi-labels', str) # single-label or multi-labels
    parser.add('train_augment', 'resize-crop-flip', str) 
    parser.add('test_augment', 'resize-crop', str) 

    parser.add('pretrained_model_classes', 16849, int)
    parser.add('pretrained_model_path', '', str)

    args = parser.get_args()
    
    ###################################################################################
    # 2. Make directories and pathes.
    ###################################################################################
    log_dir = io_utils.create_directory(f'./experiments/logs/')
    data_dir = io_utils.create_directory(f'./experiments/data/')
    model_dir = io_utils.create_directory('./experiments/models/')
    tensorboard_dir = io_utils.create_directory(f'./experiments/tensorboards/{args.tag}/')
    
    log_path = log_dir + f'{args.tag}.txt'
    data_path = data_dir + f'{args.tag}.json'
    model_path = model_dir + f'{args.tag}.pth'

    ###################################################################################
    # 3. Set the seed number and define log function. 
    ###################################################################################
    torch_utils.set_seed(args.seed)
    log_func = lambda string='': log_utils.log_print(string, log_path)
    
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    ###################################################################################
    # 4. Make transformation
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_augment_dict = {
        'mixmax_resize':augment_utils.RandomResize(args.min_image_size, args.max_image_size),
        'flip':augment_utils.RandomHorizontalFlip(),
        'resize':transforms.Resize(args.image_size, Image.BICUBIC),
        'crop':transforms.RandomCrop(args.image_size),
        'randaugment':randaugment.RandAugment(args.randaugment_n, args.randaugment_m),
        'colorjitter':transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    }

    test_augment_dict = {
        'resize':transforms.Resize(args.image_size, Image.BICUBIC),
        'crop':transforms.CenterCrop(args.image_size)
    }
    
    train_transforms = []
    for name in args.train_augment.split('-'):
        if name in train_augment_dict.keys():
            transform = train_augment_dict[name]
        else:
            raise ValueError('unrecognize name of transform ({})'.format(name))
        
        train_transforms.append(transform)
    
    test_transforms = []
    for name in args.test_augment.split('-'):
        if name in test_augment_dict.keys():
            transform = test_augment_dict[name]
        else:
            raise ValueError('unrecognize name of transform ({})'.format(name))
        
        test_transforms.append(transform)
    
    essential_transform = [
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]
    train_transform = transforms.Compose(train_transforms + essential_transform)
    test_transform = transforms.Compose(test_transforms + essential_transform)
    
    ###################################################################################
    # 5. Make datasets
    ###################################################################################
    data_dict = json_utils.read_json(f'./data/{args.dataset_name}.json')

    train_dataset = datasets.Dataset_For_Json(data_dict, 'train', args.task, train_transform)
    valid_dataset = datasets.Dataset_For_Json(data_dict, 'validation', args.task, test_transform)

    ###################################################################################
    # 6. Make loaders
    ###################################################################################
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    
    log_func('[i] The size of training set is {}'.format(len(train_dataset)))
    log_func('[i] The number of training iterations is {}'.format(len(train_loader)))
    log_func('[i] The number of evaluating iterations is {}'.format(len(valid_loader)))
    
    ###################################################################################
    # 7. Make Network
    ###################################################################################
    model = networks.Classifier(args.architecture, data_dict['num_classes'], pretrained=True)
    model.train()
    
    if torch.cuda.is_available():
        model.cuda()
    
    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(torch_utils.calculate_parameters(model)))
    log_func()
    
    # load pretrained model
    if args.pretrained_model_path != '':
        # for pixta
        pretrained_model = networks.Classifier(args.architecture, args.pretrained_model_classes, pretrained=False)
        torch_utils.load_model(pretrained_model, args.pretrained_model_path)

        torch_utils.transfer_model(pretrained_model, model, 'classifier')
        
        log_func('[i] Transfer Learning ({})'.format(args.pretrained_model_path))
        del pretrained_model
    
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
    
    ###################################################################################
    # 8. Define losses
    ###################################################################################
    if args.loss == 'ce':
        if args.task == 'multi-labels':
            loss_fn = nn.MultiLabelSoftMarginLoss().cuda()
        else:
            loss_fn = nn.CrossEntropyLoss().cuda()

    elif args.loss == 'focal':
        loss_fn = losses.Focal_Loss().cuda()
    
    ###################################################################################
    # 9. Define optimizer and scheduler
    ###################################################################################
    if args.batch_size > 256:
        args.lr = args.batch_size / 256 * args.lr

    # 1. Optimizer
    params = model.parameters()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=args.nesterov)
    
    # 2. Scheduler
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.max_epochs * 0.5), int(args.max_epochs * 0.75)], gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    #################################################################################################
    # 10. Training
    #################################################################################################
    trainer_params = {
        'model' : model,
        'optimizer' : optimizer,
        'scheduler' : scheduler,
        'loader' : train_loader,
        'amp' : args.amp,
        'tensorboard_dir' : tensorboard_dir,
        'losses' : [loss_fn],
        'log_names' : ['loss'],
        'max_epochs' : args.max_epochs,
        'data_path' : data_path
    }
    trainer = training_utils.Trainer(**trainer_params)

    evaluator_params = {
        'model' : model,
        'loader' : valid_loader,
        'class_names' : data_dict['class_names'],
        'task' : args.task,
        'amp' : args.amp
    }
    evaluator = evaluating_utils.Evaluator(**evaluator_params)

    if args.task == 'multi-labels':
        metric_name = 'C-F1'
    else:
        metric_name = 'mean_accuracy'

    best_valid_score = -1

    for epoch in range(args.max_epochs):
        # training
        lr = torch_utils.get_learning_rate_from_optimizer(optimizer)
        loss, train_time = trainer.step()

        # visualize log
        data = {
            'epoch' : epoch + 1,
            'lr' : lr,
            'loss' : loss,
            'time' : train_time
        }
        trainer.update_data('train', data)

        trainer.update_tensorboard('Training/loss', loss, epoch)
        trainer.update_tensorboard('Training/learning_rate', lr, epoch)

        log_func('[i] epoch={epoch:,}, lr={lr:.6f}, loss={loss:.4f}, time={time:.0f}sec'.format(**data))

        # evaluation
        if epoch % args.valid_ratio == 0:
            valid_score, eval_time = evaluator.step()
            
            if best_valid_score == -1 or best_valid_score < valid_score:
                best_valid_score = valid_score

                torch_utils.save_model(model, model_path, parallel=use_multi_gpu)
                # log_func('[i] save model')

            data = {
                'epoch' : epoch + 1,
                'score' : valid_score,
                'best_score' : best_valid_score,
                'time' : eval_time
            }
            trainer.update_data('validation', data)

            trainer.update_tensorboard('Evaluation/' + metric_name, valid_score, epoch)
            trainer.update_tensorboard('Evaluation/' + 'best_' + metric_name, best_valid_score, epoch)
            
            log_func('[i] epoch={epoch:,}, score={score:.2f}%, best_score={best_score:.2f}%, time={time:.0f}sec'.format(**data))
    
    print(args.tag)

