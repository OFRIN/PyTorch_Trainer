import os
import glob
import torch

import numpy as np

from PIL import Image

from tools.general.json_utils import read_json
from tools.ai.torch_utils import one_hot_embedding

class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        
        return data

class Merging_Dataset:
    def __init__(self, datasets):
        self.data_dic = {name:dataset for name, dataset in enumerate(datasets)}

        self.datasets = []
        for name in self.data_dic.keys():
            dataset = self.data_dic[name]
            for i in range(len(dataset)):
                self.datasets.append([name, i])

    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, index):
        name, index = self.datasets[index]
        return self.data_dic[name][index]

class Dataset_For_Folder(torch.data.utils.Dataset):
    def __init__(self, root_dir, domain, class_names, transform=None):
        self.transform = transform

        self.class_names = class_names
        self.classes = len(self.class_names)

        data_dir = root_dir + domain + '/'
        self.dataset = []

        for label, class_name in enumerate(class_names):
            dataset_per_class_name = []
            image_dir = data_dir + class_name + '/'

            for extension in ['.jpg', '.jpeg', '.png']:
                dataset_per_class_name += [[image_path, label] for image_path in glob.glob(image_dir + '*' + extension)]
            
            if len(dataset_per_class_name) != len(os.path.isdir(image_dir)):
                print('[{}] miss match : {} / {}'.format(class_name, len(dataset_per_class_name, len(os.path.isdir(image_dir)))))
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, label = self.dataset[index]

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label
