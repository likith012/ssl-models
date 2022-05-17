#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#1d conv block
def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)


class Bottleneck1d(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet1d(nn.Module):
    '''1d adaptation of the torchvision resnet'''
    def __init__(self, block, layers, kernel_size=3, input_channels=1, inplanes=64, fix_feature_dim=False, kernel_size_stem = 7, stride_stem=2, pooling_stem=False, stride=1):

        super(ResNet1d,self).__init__()

        self.inplanes = inplanes
        self.layers = nn.ModuleList([])
        if(kernel_size_stem is None):
            kernel_size_stem = kernel_size[0] if isinstance(kernel_size,list) else kernel_size

        #stem
        self.layers.append(nn.Conv1d(input_channels, inplanes, kernel_size=kernel_size_stem, stride=stride_stem, padding=(kernel_size_stem-1)//2,bias=False))
        self.layers.append(nn.BatchNorm1d(inplanes))
        self.layers.append(nn.ReLU(inplace=True))

        if(pooling_stem is True):
            self.layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        #backbone
        for i,l in enumerate(layers):

            if(i==0):
                self.layers.append(self._make_layer(block, inplanes, layers[0],kernel_size=kernel_size))

            else:
                self.layers.append(self._make_layer(block, inplanes if fix_feature_dim else (2**i)*inplanes, layers[i], stride=2,kernel_size=kernel_size))
        

    def _make_layer(self, block, planes, blocks, stride=1,kernel_size=3):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            print('downsample')
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel_size, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        for lay in self.layers:
            x = lay(x)
        return x


def resnet1d50(config,**kwargs):
    """Constructs a ResNet-50 model.
    """
    return ResNet1d(Bottleneck1d, [3, 4, 6, 3],kernel_size=config.kernel_size,stride=config.stride,kernel_size_stem=config.kernel_size_stem,stride_stem=config.stride_stem,**kwargs)

