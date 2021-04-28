import glob
import torch
import numpy as np

from tools.data.reader import SH_Dataset
from tools.data.utils import decode_image
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

class Dataset_For_Visualization(SH_Dataset):
    def __init__(self, data_dir, domain):
        super().__init__(data_dir + f'{domain}/*.sang', None, debug=True)
        
    def decode(self, example):
        image = decode_image(example['image'])

        human_label = example['human_tags']
        google_label = example['google_tags']

        return image, human_label, google_label

class Dataset_For_Folder(torch.data.utils.Dataset):
    def __init__(self, )

class Dataset_For_OGQ_3M(SH_Dataset):
    def __init__(self, data_dir, domain, data_dic, transform):
        super().__init__(data_dir + f'{domain}/*.sang', transform, debug=True)
        
        self.class_names = np.asarray(data_dic['class_names'])
        self.class_dic = {name : index for index, name in enumerate(self.class_names)}
        self.classes = data_dic['classes']
    
    def decode(self, example):
        image = decode_image(example['image'])
        if self.transform is not None:
            image = self.transform(image)

        label = one_hot_embedding([self.class_dic[name] for name in example['google_tags'] if name in self.class_names], self.classes)
        return image, label
