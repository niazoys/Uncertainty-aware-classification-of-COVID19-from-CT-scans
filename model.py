import sys
import torch
import utils
import torch.nn as nn
from torchvision import models
from utils import keep_variance
from typing import Union, List, Dict, Any, cast
from adf_blocks import Conv2d,AvgPool2d,ReLU,Dropout,Linear,MaxPool2d,BatchNorm2d,Sequential

class vgg19(nn.Module):
    
    def __init__(self, n_classes, pretrained=False):    
        super(vgg19, self).__init__()
        self.n_classes = n_classes
        self.model = models.vgg16_bn(pretrained=pretrained,num_classes=self.n_classes)
        # self.fc=nn.Linear(1000,n_classes)

    def forward(self, x):
        x=self.model(x)
        # x=self.fc(x)
        return x

class resnet18(nn.Module):
    
    def __init__(self, n_classes, pretrained=True):    
        super(resnet18, self).__init__()
        self.n_classes = n_classes
        self.model = models.resnet18(pretrained=pretrained,num_classes= self.n_classes)
        # self.fc=nn.Linear(1000,n_classes)

    def forward(self, x):
        x=self.model(x)
        # x=self.fc(x)
        return x

class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        drop_probability :float = 0.2,
        min_variance: float =1e-3,
        noise_variance: float =1e-3
    ) -> None:
        super(VGG, self).__init__()
        self.var_fun = lambda x: keep_variance(x, min_variance=min_variance)
        self.features = features
        self._noise_variance = noise_variance
        self.avgpool = AvgPool2d(keep_variance_fn=self.var_fun,kernel_size=4)
        self.classifier = Sequential(
            Linear(in_features= 512 , out_features=512 , keep_variance_fn=self.var_fun),
            ReLU(keep_variance_fn=self.var_fun),
            Dropout(p=drop_probability,keep_variance_fn=self.var_fun),
            # Linear(in_features=4096,out_features=4096,keep_variance_fn=self.var_fun),
            # ReLU(keep_variance_fn=self.var_fun),
            # Dropout(p=drop_probability,keep_variance_fn=self.var_fun),
            Linear(in_features=512,out_features= num_classes,keep_variance_fn=self.var_fun),
        )
   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs_mean = x
        inputs_variance = torch.zeros_like(inputs_mean) + self._noise_variance
        x = (inputs_mean, inputs_variance)
        x = self.features(*x)
        x = self.avgpool(*x)
        x_mean = torch.flatten(x[0], 1)
        x_var = torch.flatten(x[1], 1)
        x=x_mean,x_var
        x = self.classifier(*x)
        return x
    
def make_layers(cfg: List[Union[str, int]],dropout_prob,min_variance: float =1e-3,input_channel: int= 3) -> Sequential:
    layers: List[nn.Module] = []
    var_fun=lambda x:keep_variance(x,min_variance=min_variance)
    in_channels=input_channel
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(keep_variance_fn=var_fun)]
        else:
            v = cast(int, v)
            conv2d = Conv2d(in_channels, v,kernel_size=3,padding=1,keep_variance_fn=var_fun)
            layers += [conv2d, BatchNorm2d(v,keep_variance_fn=var_fun), ReLU(keep_variance_fn=var_fun),Dropout(p=dropout_prob,keep_variance_fn=var_fun)]
            in_channels = v

    return Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg_adf(variant: str='vgg16',input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any) -> VGG:
    if variant == 'vgg19':
        model = VGG(make_layers(cfgs['E'],input_channel=input_channel,min_variance=kwargs['min_variance'],dropout_prob=dropout_prob), **kwargs)
    elif variant == 'vgg16':
        model = VGG(make_layers(cfgs['D'],input_channel=input_channel,min_variance=kwargs['min_variance'],dropout_prob=dropout_prob), **kwargs)
    elif variant == 'vgg13':
        model = VGG(make_layers(cfgs['B'],input_channel=input_channel,min_variance=kwargs['min_variance'],dropout_prob=dropout_prob), **kwargs)
    elif variant == 'vgg11':
        model = VGG(make_layers(cfgs['A'],input_channel=input_channel,min_variance=kwargs['min_variance'],dropout_prob=dropout_prob), **kwargs)
    else:
        sys.exit("Unknown variant")
    return model

def vgg(variant: str='vgg16',**kwargs: Any) -> VGG:
    if variant == 'vgg19':
        model = models.vgg19_bn(pretrained=False,num_classes=kwargs['num_classes'])
    elif variant == 'vgg16':
         model = models.vgg16_bn(pretrained=False,num_classes=kwargs['num_classes'])
    elif variant == 'vgg13':
         model = models.vgg13_bn(pretrained=False,num_classes=kwargs['num_classes'])
    elif variant == 'vgg11':
        model = models.vgg11_bn(pretrained=False,num_classes=kwargs['num_classes'])
    else:
        sys.exit("Unknown variant")
    return model