import sys
from typing import Any
from torchvision import models


def resnet152(**kwargs: Any)  -> models.ResNet :
    return models.resnet152(pretrained=False,num_classes=kwargs['num_classes'])  
def resnet101(**kwargs: Any)  -> models.ResNet :
    return models.resnet101(pretrained=False,num_classes=kwargs['num_classes'])  
def resnet50(**kwargs: Any)  -> models.ResNet :
    return models.resnet50(pretrained=False,num_classes=kwargs['num_classes'])  
def resnet34(**kwargs: Any)  -> models.ResNet :
    return models.resnet34(pretrained=False,num_classes=kwargs['num_classes'])     
def resnet18(**kwargs: Any)  -> models.ResNet :
    return models.resnet18(pretrained=False,num_classes=kwargs['num_classes'])    


def vgg19(**kwargs: Any) -> models.VGG:
    return models.vgg19_bn(pretrained=False,num_classes=kwargs['num_classes'])
def vgg16(**kwargs: Any) -> models.VGG:
    return models.vgg16_bn(pretrained=False,num_classes=kwargs['num_classes'])
def vgg13(**kwargs: Any) -> models.VGG:
    return models.vgg13_bn(pretrained=False,num_classes=kwargs['num_classes'])
def vgg11(**kwargs: Any) -> models.VGG:
    return models.vgg11_bn(pretrained=False,num_classes=kwargs['num_classes'])
  