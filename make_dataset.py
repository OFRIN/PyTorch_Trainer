# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import cv2
import sys

from core import datasets
from tools.general import io_utils, json_utils

if __name__ == '__main__':
    ###################################################################################
    # 1. Arguments
    ###################################################################################
    parser = io_utils.Parser()

    parser.add('json_path', './data/leak.json', str)
    parser.add('data_dir', 'D:/Leak_Detection/', str)

    parser.add('task', 'multi-labels', str)
    parser.add('class_names', '', str)

    parser.add('domains', 'train,validation,test', str)
    parser.add('demo', True, bool)
    
    args = parser.get_args()
    
    ###################################################################################
    # 2. Make datasets
    ###################################################################################
    domains = args.domains.split(',')
    class_names = args.class_names.split(',')

    print(domains, class_names)
    
    if args.task == 'single-label':
        dataset_dict = {domain:datasets.Dataset_For_Folder(args.data_dir, domain, class_names) for domain in domains}
    
    ###################################################################################
    # 3. Check datasets
    ###################################################################################
    data_dict = {}

    data_dict['class_names'] = class_names
    data_dict['num_classes'] = len(class_names)
    data_dict['class_dict'] = {name:index for index,name in enumerate(class_names)}

    for domain in domains:
        data_dict[domain] = []

        dataset = dataset_dict[domain]
        length = len(dataset)

        digits = io_utils.get_digits_in_number(length)

        for i, (image_path, label) in enumerate(dataset):
            i += 1
            progress_format = '\r[%0{}d/%0{}d] = %02.2f%%'.format(digits, digits)

            sys.stdout.write(progress_format%(i, length, i / length * 100))
            sys.stdout.flush()

            if image_path is None:
                continue
            
            # quick demo
            if args.demo:
                image = cv2.imread(image_path)
                print(label)

                cv2.imshow('image', image)
                cv2.waitKey(0)

            data_dict[domain].append([image_path, label])
        print()

    json_utils.write_json(args.json_path, data_dict, encoding='utf-8')