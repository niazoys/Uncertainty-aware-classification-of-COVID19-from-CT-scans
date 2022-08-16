import sys
import torch
import adf_blocks 
import torch.nn as nn
from utils import keep_variance
from typing import Union, List, Dict, Any, cast


########## VGG Architectures #########

class VGGADF(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        drop_probability :float = 0.2,
        min_variance: float =1e-3,
        noise_variance: float =1e-3
    ) -> None:
        super(VGGADF, self).__init__()
        self.var_fun = lambda x: keep_variance(x, min_variance=min_variance)
        self.features = features
        self._noise_variance = noise_variance
        self.avgpool = adf_blocks.AvgPool2d(keep_variance_fn=self.var_fun,kernel_size=2)
        self.classifier = adf_blocks.Sequential(
            adf_blocks.Linear(in_features= 512*4 , out_features=512 , keep_variance_fn=self.var_fun),
            adf_blocks.ReLU(keep_variance_fn=self.var_fun),
            adf_blocks.Dropout(p=drop_probability,keep_variance_fn=self.var_fun),
            # Linear(in_features=4096,out_features=4096,keep_variance_fn=self.var_fun),
            # ReLU(keep_variance_fn=self.var_fun),
            # Dropout(p=drop_probability,keep_variance_fn=self.var_fun),
            adf_blocks.Linear(in_features=512,out_features= num_classes,keep_variance_fn=self.var_fun),
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
    
def make_layers(cfg: List[Union[str, int]],dropout_prob,min_variance: float =1e-5,input_channel: int= 3) -> adf_blocks.Sequential:
    layers: List[nn.Module] = []
    var_fun=lambda x:keep_variance(x,min_variance=min_variance)
    in_channels=input_channel
    for v in cfg:
        if v == 'M':
            layers += [adf_blocks.MaxPool2d(keep_variance_fn=var_fun)]
        else:
            v = cast(int, v)
            conv2d = adf_blocks.Conv2d(in_channels, v,kernel_size=3,padding=1,keep_variance_fn=var_fun)
            layers += [conv2d, adf_blocks.BatchNorm2d(v,keep_variance_fn=var_fun), adf_blocks.ReLU(keep_variance_fn=var_fun),adf_blocks.Dropout(p=dropout_prob,keep_variance_fn=var_fun)]
            in_channels = v

    return adf_blocks.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

########## ResNet Architectures #########

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.2, keep_variance_fn=None):
        super(BasicBlock, self).__init__()
        
        self.keep_variance_fn = keep_variance_fn
        
        self.conv1 = adf_blocks.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn1 = adf_blocks.BatchNorm2d(planes, keep_variance_fn=self.keep_variance_fn)
        self.conv2 = adf_blocks.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn2 = adf_blocks.BatchNorm2d(planes, keep_variance_fn=self.keep_variance_fn)
        self.ReLU = adf_blocks.ReLU(keep_variance_fn=self.keep_variance_fn)

        self.shortcut = adf_blocks.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = adf_blocks.Sequential(
                adf_blocks.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, keep_variance_fn=self.keep_variance_fn),
                adf_blocks.BatchNorm2d(self.expansion*planes, keep_variance_fn=self.keep_variance_fn)
            )
            
        self.dropout = adf_blocks.Dropout(p=p, keep_variance_fn=self.keep_variance_fn)

    def forward(self, inputs_mean, inputs_variance):
        x = inputs_mean, inputs_variance
        
        out = self.dropout(*self.ReLU(*self.bn1(*self.conv1(*x))))
        out_mean, out_var = self.bn2(*self.conv2(*out))
        shortcut_mean, shortcut_var = self.shortcut(*x)
        out_mean, out_var = out_mean + shortcut_mean, out_var + shortcut_var
        out = out_mean, out_var 
        out = self.dropout(*self.ReLU(*out))
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, p=0.2, keep_variance_fn=None):
        super(Bottleneck, self).__init__()
        
        self.keep_variance_fn = keep_variance_fn
        
        self.conv1 = adf_blocks.Conv2d(in_planes, planes, kernel_size=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn1 = adf_blocks.BatchNorm2d(planes, keep_variance_fn=self.keep_variance_fn)
        self.conv2 = adf_blocks.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn2 = adf_blocks.BatchNorm2d(planes, keep_variance_fn=self.keep_variance_fn)
        self.conv3 = adf_blocks.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn3 = adf_blocks.BatchNorm2d(self.expansion*planes, keep_variance_fn=self.keep_variance_fn)
        self.ReLU = adf_blocks.ReLU(keep_variance_fn=self.keep_variance_fn)

        self.shortcut = adf_blocks.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = adf_blocks.Sequential(
                adf_blocks.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, keep_variance_fn=self.keep_variance_fn),
                adf_blocks.BatchNorm2d(self.expansion*planes, keep_variance_fn=self.keep_variance_fn)
            )
            
        self.dropout = adf_blocks.Dropout(p=p, keep_variance_fn=self.keep_variance_fn)

    def forward(self, inputs_mean, inputs_variance):
        x = inputs_mean, inputs_variance
        
        out = self.dropout(*self.ReLU(*self.bn1(*self.conv1(*x))))
        out = self.dropout(*self.ReLU(*self.bn2(*self.conv2(*out))))
        out = self.bn3(*self.conv3(*out))
        out += self.shortcut(*x)
        out = self.dropout(*self.ReLU(*out))
        return out

class ResNetADF(nn.Module):
    def __init__(self, block, num_blocks,input_channel=3, num_classes=10, p=0.2, noise_variance=1e-3, min_variance=1e-3):
        super(ResNetADF, self).__init__()

        self.keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._noise_variance = noise_variance

        self.in_planes = 64

        self.conv1 = adf_blocks.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn1 = adf_blocks.BatchNorm2d(64, keep_variance_fn=self.keep_variance_fn)
        self.ReLU = adf_blocks.ReLU(keep_variance_fn=self.keep_variance_fn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, p=p, keep_variance_fn=self.keep_variance_fn)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, p=p, keep_variance_fn=self.keep_variance_fn)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, p=p, keep_variance_fn=self.keep_variance_fn)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, p=p, keep_variance_fn=self.keep_variance_fn)
        self.linear = adf_blocks.Linear(512*block.expansion, num_classes, keep_variance_fn=self.keep_variance_fn)
        self.AvgPool2d = adf_blocks.AvgPool2d(keep_variance_fn=self.keep_variance_fn)
        
        self.dropout = adf_blocks.Dropout(p=p, keep_variance_fn=self.keep_variance_fn)

    def _make_layer(self, block, planes, num_blocks, stride, p=0.2, keep_variance_fn=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p=p, keep_variance_fn=self.keep_variance_fn))
            self.in_planes = planes * block.expansion
        return adf_blocks.Sequential(*layers)

    def forward(self, x):
 
        inputs_mean = x
        inputs_variance = torch.zeros_like(inputs_mean) + self._noise_variance
        x = inputs_mean, inputs_variance

        out = self.dropout(*self.ReLU(*self.bn1(*self.conv1(*x))))
        out = self.layer1(*out)
        out = self.layer2(*out)
        out = self.layer3(*out)
        out = self.layer4(*out)
        out = self.AvgPool2d(*out, 9)
        out_mean = out[0].view(out[0].size(0), -1) 
        out_var = out[1].view(out[1].size(0), -1)
        out = out_mean, out_var
        out = self.linear(*out)
        return out

cfgs_resnet: Dict[ str, List[Union[nn.Module,int]] ] = {
    'A': [ BasicBlock,[2,2,2,2]  ],
    'B': [ BasicBlock,[3,4,6,3]  ],
    'C': [ Bottleneck,[3,4,6,3]  ],
    'D': [ Bottleneck,[3,4,23,3] ],
    'E': [ Bottleneck,[3,8,36,3] ],
}

####### Model construction methods ##########

def vgg19(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any) -> VGGADF:
    return VGGADF(make_layers(cfgs['E'],input_channel=input_channel,dropout_prob=dropout_prob), dropout=dropout_prob,**kwargs)
def vgg16(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any) -> VGGADF: 
    return VGGADF(make_layers(cfgs['D'],input_channel=input_channel,dropout_prob=dropout_prob),dropout=dropout_prob, **kwargs)
def vgg13(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any) -> VGGADF:
    return VGGADF(make_layers(cfgs['B'],input_channel=input_channel,dropout_prob=dropout_prob),dropout=dropout_prob, **kwargs)
def vgg11(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any) -> VGGADF:
    return VGGADF(make_layers(cfgs['A'],input_channel=input_channel,dropout_prob=dropout_prob),dropout=dropout_prob, **kwargs)
    
def resnet152(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any)  -> ResNetADF :
    return ResNetADF(cfgs_resnet['E'][0],cfgs_resnet['E'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],p=dropout_prob,min_variance=kwargs['min_variance'],noise_variance=kwargs['noise_variance'])    
def resnet101(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any)  -> ResNetADF :
    return ResNetADF(cfgs_resnet['D'][0],cfgs_resnet['D'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],p=dropout_prob,min_variance=kwargs['min_variance'],noise_variance=kwargs['noise_variance'])    
def resnet50(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any)  -> ResNetADF :
    return ResNetADF(cfgs_resnet['C'][0],cfgs_resnet['C'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],p=dropout_prob,min_variance=kwargs['min_variance'],noise_variance=kwargs['noise_variance'])    
def resnet34(input_channel:int=3, dropout_prob:float=0.2 ,**kwargs: Any)  -> ResNetADF :
    return ResNetADF(cfgs_resnet['B'][0],cfgs_resnet['B'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],p=dropout_prob,min_variance=kwargs['min_variance'],noise_variance=kwargs['noise_variance'])    
def resnet18(input_channel:int=3, dropout_prob:float=0.2 , **kwargs: Any)  -> ResNetADF :
    return ResNetADF(cfgs_resnet['A'][0],cfgs_resnet['A'][1],input_channel=input_channel,num_classes=kwargs['num_classes'],p=dropout_prob,min_variance=kwargs['min_variance'],noise_variance=kwargs['noise_variance'])    

