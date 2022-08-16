import sys
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast


########## VGG Architectures #########

class VGGDropout(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg: List[Union[str, int]],dropout_prob=0.2) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
           
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True),nn.Dropout(dropout_prob)]
           
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

########## ResNet Architectures #######

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.2):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        out = self.dropout(nn.functional.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout(nn.functional.relu(out))
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        out = self.dropout(nn.functional.relu(self.bn1(self.conv1(x))))
        out = self.dropout(nn.functional.relu(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.dropout(nn.functional.relu(out))
        return out

class ResNetDropout(nn.Module):
    def __init__(self, block,num_blocks, input_channel=3,num_classes=10, dropout_prob=0.2):
        super(ResNetDropout, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,dropout_prob=dropout_prob)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.dropout = nn.Dropout2d(dropout_prob)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_prob=0.2):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_prob=dropout_prob))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dropout(nn.functional.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

cfgs_resnet: Dict[ str, List[Union[nn.Module,int]] ] = {
    'A': [ BasicBlock,[2,2,2,2]  ],
    'B': [ BasicBlock,[3,4,6,3]  ],
    'C': [ Bottleneck,[3,4,6,3]  ],
    'D': [ Bottleneck,[3,4,23,3] ],
    'E': [ Bottleneck,[3,8,36,3] ],
}

####### Model construction methods #####

def vgg19(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any) -> VGGDropout:
    return VGGDropout(make_layers(cfgs['E'],input_channel=input_channel,dropout_prob=dropout_prob), dropout=dropout_prob,**kwargs)
def vgg16(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any) -> VGGDropout: 
    return VGGDropout(make_layers(cfgs['D'],input_channel=input_channel,dropout_prob=dropout_prob),dropout=dropout_prob, **kwargs)
def vgg13(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any) -> VGGDropout:
    return VGGDropout(make_layers(cfgs['B'],input_channel=input_channel,dropout_prob=dropout_prob),dropout=dropout_prob, **kwargs)
def vgg11(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any) -> VGGDropout:
    return VGGDropout(make_layers(cfgs['A'],input_channel=input_channel,dropout_prob=dropout_prob),dropout=dropout_prob, **kwargs)
    
def resnet152(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any)  ->  ResNetDropout :
    return ResNetDropout(cfgs_resnet['E'][0],cfgs_resnet['E'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],dropout_prob=dropout_prob)  
def resnet101(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any)  ->  ResNetDropout :
    return ResNetDropout(cfgs_resnet['D'][0],cfgs_resnet['D'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],dropout_prob=dropout_prob)  
def resnet50(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any)  ->   ResNetDropout :
    return ResNetDropout(cfgs_resnet['C'][0],cfgs_resnet['C'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],dropout_prob=dropout_prob)  
def resnet34(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any)  ->   ResNetDropout :
    return ResNetDropout(cfgs_resnet['B'][0],cfgs_resnet['B'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],dropout_prob=dropout_prob)  
def resnet18(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any)  ->   ResNetDropout :
    return ResNetDropout(cfgs_resnet['A'][0],cfgs_resnet['A'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],dropout_prob=dropout_prob)  

