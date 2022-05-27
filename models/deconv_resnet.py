#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import (
    conv3x3,
    conv1x1,
)

def deconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    ''' 3x3 convolution Transpose with padding '''
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                              output_padding=dilation, groups=groups, bias=False, dilation=dilation)

def deconv1x1(in_planes, out_planes, stride=1, output_padding=1):
    ''' 1x1 convolution Transpose '''
    return nn.ConvTranspose2d(
        in_planes, out_planes, kernel_size=1,
        stride=stride, bias=False, output_padding=output_padding)


class DeconvBasicBlock(nn.Module):
    
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int=1,
        sample: nn.Module=None,
        norm_layer: nn.Module=None,
        groups: int=1,
        base_width: int=64,
        dilation: int=1,
    ) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        if groups != 1 or base_width != 64:
            raise ValueError('DeconvBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
            
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        if stride == 1:
            self.conv2 = conv3x3(planes, planes, stride)
        else:
            self.conv2 = deconv3x3(planes, planes, stride)
        
        self.bn2 = norm_layer(planes)
        self.sample = sample
        self.stride = stride

    def forward(
        self,
        x: torch.Tensor, 
    ) -> torch.Tensor:
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.sample is not None:
            identity = self.sample(x)

        out += identity
        out = self.relu(out)

        return out
    
class DeconvBottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int=1,
        sample: nn.Module=None,
        groups: int=1,
        base_width: int=64,
        dilation: int=1, 
        norm_layer: nn.Module=None,
    ) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        mid = inplanes // self.expansion
            
        width = int(mid * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        if stride == 1:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        else:
            self.conv2 = deconv3x3(width, width, stride, groups, dilation)
            
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.sample = sample

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.sample is not None:
            print('lol')
            shortcut = self.sample(x)

        out += shortcut
        print('kek')
        out = self.relu(out)

        return out
    
    
class DeconvResNet(nn.Module):
    
    def __init__(
        self,
        block: nn.Module,
        layers: list,
        out_channels: int=3,
        zero_init_residual: bool=True,
        groups: int=1,
        width_per_group: int=64,
        replace_stride_with_dilation: list=None,
        norm_layer: nn.Module=None,
    ) -> None:
        
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self._norm_layer = norm_layer
        
        self.dilation = 1
        self.inplanes = 512 * block.expansion
        
        self.groups = groups
        self.base_width = width_per_group
        self.zero_init_residual = zero_init_residual
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None '
                f'or a 3 element tuple, got {replace_stride_with_dilation}'
            )
            
        # decoder

        self.uplayer1 = self._make_layer(
            block, 256, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])
        self.uplayer2 = self._make_layer(
            block, 128, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.uplayer3 = self._make_layer(
            block, 64, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.uplayer4 = self._make_layer(
            block, 64, layers[0])
        
        sample = nn.Sequential(
            deconv1x1(64, 64, stride=2),
            norm_layer(64),
        )
        self.uplayer_top = block(
            64, planes=64, stride=2, sample=sample)

        self.conv1_1 = deconv3x3(64, out_channels, stride=2)
            
        self.init_weights()
        
    def init_weights(
        self,
    ) -> None:
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    tensor=m.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, DeconvBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, DeconvBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        blocks: int,
        stride: int=1,
        dilate: bool=False,
    ) -> nn.Module:
        
        unsample = None
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes:
            unsample = nn.Sequential(
                deconv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, self.inplanes // block.expansion,
                    stride=1,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
            
        layers.append(
            block(
                self.inplanes, planes,
                stride=stride,
                sample=unsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
            )
        )
        
        self.inplanes = self.inplanes // 2

        return nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        
        x = self.uplayer_top(x)

        x = self.conv1_1(x)

        return x
    
def _deconv_resnet(
    arch: str,
    block: nn.Module,
    layers: list,
    **kwargs,
) -> DeconvResNet:
    
    model = DeconvResNet(block, layers, **kwargs)
    return model

def deconv_resnet18(
    **kwargs,
) -> DeconvResNet:
    
    r'''ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    '''
    return _deconv_resnet('deconv_resnet18', DeconvBasicBlock, [2, 2, 2, 2], **kwargs)

def deconv_resnet34(
    **kwargs,
) -> DeconvResNet:
    
    r'''ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    '''
    return _deconv_resnet('deconv_resnet34', DeconvBasicBlock, [3, 4, 6, 3], **kwargs)

def deconv_resnet50(
    **kwargs,
) -> DeconvResNet:
    
    r'''ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    '''
    return _deconv_resnet('deconv_resnet50', DeconvBottleneck, [3, 4, 6, 3], **kwargs)

def deconv_resnet101(
    **kwargs,
) -> DeconvResNet:
    
    r'''ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    '''
    return _deconv_resnet('deconv_resnet101', DeconvBottleneck, [3, 4, 23, 3], **kwargs)
        
        