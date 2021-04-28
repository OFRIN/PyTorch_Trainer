
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from .efficientnet_pytorch.model import EfficientNet

class Tagging(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        
        if 'ghostnet' in model_name:
            self.model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=pretrained)
            
            self.in_channels = 960
            self.features = nn.Sequential(*list(self.model.children())[:-4])
        
        elif 'mobilenetv2' in model_name:
            self.model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=pretrained)
            
            self.in_channels = 1280
            self.features = nn.Sequential(*list(self.model.children())[:-1])
        
        elif 'efficientnet' in model_name:
            if pretrained:
                self.model = EfficientNet.from_pretrained(model_name)
            else:
                self.model = EfficientNet.from_name(model_name)
            
            # self.in_channels = 1280
            self.in_channels = 2048
            self.features = self.model

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(self.model._global_params.dropout_rate)
        self.classifier = nn.Conv2d(self.in_channels, num_classes, 1)

        self.initialize([self.classifier])
    
    def global_average_pooling_2d(self, x, keepdims=False):
        x = self.avg_pool(x)
        if not keepdims:
            x = x.view(x.size(0), x.size(1))
        return x
    
    def forward(self, x):
        b = x.size()[0]
        x = self.features(x)

        x = self.global_average_pooling_2d(x, keepdims=True)

        x = self.dropout(x)
        logits = self.classifier(x).view(b, -1)
        return logits
    
    def initialize(self, weights):
        for m in weights:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def get_parameter_groups(self, print_fn=print):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'model' in name or 'features' in name:
                if 'weight' in name:
                    # print_fn(f'pretrained weights : {name}')
                    groups[0].append(value)
                else:
                    # print_fn(f'pretrained bias : {name}')
                    groups[1].append(value)
                    
            # scracthed weights
            else:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'scratched weights : {name}')
                    groups[2].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'scratched bias : {name}')
                    groups[3].append(value)
        return groups
