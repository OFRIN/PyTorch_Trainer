
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from .architectures import efficientnet 
from .architectures import resnest 

from abc import ABC

class ABC_Model(ABC):
    def initialize(self, weights):
        for m in weights:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
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

class Classifier(nn.Module, ABC_Model):
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
                self.model = efficientnet.model.EfficientNet.from_pretrained(model_name)
            else:
                self.model = efficientnet.model.EfficientNet.from_name(model_name)
            
            if 'b0' in model_name:
                self.in_channels = 1280
            else:
                self.in_channels = 2048
            
            self.features = self.model
        
        elif 'resnest' in model_name:
            # ex. model_name = resnest101
            self.model = eval("resnest." + model_name)(pretrained=True, dilated=2, dilation=False)

            del self.model.avgpool
            del self.model.fc
            
            stage1 = nn.Sequential(self.model.conv1, 
                                   self.model.bn1, 
                                   self.model.relu, 
                                   self.model.maxpool)
            stage2 = nn.Sequential(self.model.layer1)
            stage3 = nn.Sequential(self.model.layer2)
            stage4 = nn.Sequential(self.model.layer3)
            stage5 = nn.Sequential(self.model.layer4)

            self.in_channels = 2048
            self.features = nn.Sequential(
                stage1,
                stage2,
                stage3,
                stage4,
                stage5
            )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Conv2d(self.in_channels, num_classes, 1)

        self.initialize([self.classifier])

    def global_average_pooling_2d(self, x, keepdims=False):
        x = self.avg_pool(x)
        if not keepdims:
            x = x.view(x.size(0), x.size(1))
        return x

    def make_cams(self, x):
        x = F.relu(x)
        x = x / (F.adaptive_max_pool2d(x, (1, 1)) + 1e-5)
        return x
    
    def forward(self, x):
        b = x.size()[0]
        x = self.features(x)

        x = self.global_average_pooling_2d(x, keepdims=True)

        logits = self.classifier(x).view(b, -1)
        return logits

    def forward_with_cam(self, x):
        b = x.size()[0]
        x = self.features(x)
        f = self.classifier(x)
        
        x = self.global_average_pooling_2d(f, keepdims=True)
        logits = x.view(b, -1)
        
        return logits, self.make_cams(f)

    
