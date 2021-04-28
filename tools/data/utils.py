# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2

import pickle
import numpy as np

from io import BytesIO
from PIL import Image

def deserialize(data):
    return pickle.loads(data)

def serialize(data):
    return pickle.dumps(data)

def encode_image(image_data, version='pillow', mode='png'):
    # pillow version
    if version == 'pillow':
        buffer = BytesIO()

        if mode == 'png':
            image_data.save(buffer, format='PNG')
        else:
            image_data.save(buffer, format='JPEG', subsampling=0, quality=85)
            
        return buffer

    # opencv version
    else:
        _, image_data = cv2.imencode('.jpg', image_data)
        return image_data

def decode_image(image_data, version='pillow'):
    # pillow version
    if version == 'pillow':
        return Image.open(image_data)

    # opencv version
    else:
        image_data = np.fromstring(image_data, dtype = np.uint8)
        image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return image_data

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

def get_labels_from_dataset(dataset):
    try:
        labels = dataset.targets
    except:
        labels = [label for image, label in dataset]
    
    return np.asarray(labels)

def split_train_and_validation_datasets(dataset, classes, ratio=0.1):
    labels = get_labels_from_dataset(dataset)

    train_indices = []
    validation_indices = []

    for class_index in range(classes):
        indices = np.where(labels == class_index)[0]
        validation_size_per_class = int(len(indices) * ratio)
        
        np.random.shuffle(indices)
        
        train_indices.extend(indices[:-validation_size_per_class])
        validation_indices.extend(indices[-validation_size_per_class:])
    
    return train_indices, validation_indices

