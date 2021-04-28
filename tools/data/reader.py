# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import sys
import glob
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader

from tools.data.utils import *

class SH_Dataset(Dataset):
    def __init__(self, data_pattern, transform, debug=False):
        if isinstance(data_pattern, str):
            self.data_paths = glob.glob(data_pattern)
        else:
            self.data_paths = data_pattern
            
        self.dataset = []
        self.transform = transform
        
        length = len(self.data_paths)
        for index, path in enumerate(self.data_paths):
            if debug: 
                sys.stdout.write(f'\r [{index+1}/{length}] ' + path.replace('.sang', '.index'))
                sys.stdout.flush()

            fp = open(path.replace('.sang', '.index'), 'r')

            for string in fp.readlines():
                start_point, length_of_example = string.strip().split(',')
                self.dataset.append((path, int(start_point), int(length_of_example)))
            
            fp.close()
        
        if debug:
            print()
    
    def __len__(self):
        return len(self.dataset)

    def decode(self, example):
        image = decode_image(example['encoded_image'])
        if self.transform is not None:
            image = self.transform(image)

        label = example['label']
        return image, label

    def get_example(self, data):
        path, start_point, length_of_example = data
        
        fp = open(path, 'rb')

        fp.seek(start_point, 1)
        bytes_of_example = fp.read(length_of_example)

        fp.close()

        return deserialize(bytes_of_example)
    
    def __getitem__(self, index):
        example = self.get_example(self.dataset[index])
        return self.decode(example)

