import os
import glob
import torch

from PIL import Image

from tools.ai import torch_utils

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

class Dataset_For_Folder(torch.utils.data.Dataset):
    def __init__(self, 
        root_dir, domain, class_names,
        extensions=['.jpg', '.jpeg', '.png', '.webp', '.gif']):

        self.class_names = class_names
        self.classes = len(self.class_names)

        data_dir = root_dir + domain + '/'
        self.dataset = []

        for class_name in class_names:
            dataset_per_class_name = []
            image_dir = data_dir + class_name + '/'

            for extension in extensions:
                dataset_per_class_name += [[image_path, [class_name]] for image_path in glob.glob(image_dir + '*' + extension) if len(image_path) < 260]
            
            self.dataset += dataset_per_class_name
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image_path, label = self.dataset[index]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = image_path
        except:
            image = None

        return image, label

class Dataset_For_Json(torch.utils.data.Dataset):
    def __init__(self, data_dict, domain, task, transform=None):
        self.task = task
        self.transform = transform

        self.dataset = data_dict[domain]

        self.class_dict = data_dict['class_dict']
        self.num_classes = data_dict['num_classes']
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, class_names = self.dataset[index]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if self.task == 'multi-labels':
            label = torch_utils.one_hot_embedding([self.class_dict[name] for name in class_names], self.num_classes)
        else:
            label = self.class_dict[class_names[0]]

        return image, label