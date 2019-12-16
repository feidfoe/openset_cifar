from __future__ import absolute_import

import torch.nn as nn
import math
import torch

__all__ = ['ae']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class AE(nn.Module):

    def __init__(self, depth, num_classes=1000, 
                       block_name='BasicBlock'):
        super(AE, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.init_relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.bn1d   = nn.BatchNorm1d(64 * block.expansion,
                                     affine=False)
        self.fc = nn.Linear(64 * block.expansion, 
                            num_classes,
                            bias=False)

        self.ae_conv1   = nn.Conv2d(64*block.expansion, 64,
                                  kernel_size=3,
                                  bias=False,
                                  padding=1)
        self.ae_bn1     = nn.BatchNorm2d(64)
        self.ae_conv2   = nn.Conv2d(64, 64,
                                    kernel_size=3,
                                    bias=False,
                                    padding=1)
        self.ae_bn2     = nn.BatchNorm2d(64)
        self.ae_deconv1 = nn.ConvTranspose2d(64, 32,
                                             bias=False,
                                             stride=2,
                                             padding=1,
                                             kernel_size=4)
        self.ae_bn_d1   = nn.BatchNorm2d(32)
        self.ae_conv3   = nn.Conv2d(32, 32,
                                  kernel_size=3,
                                  bias=False,
                                  padding=1)
        self.ae_bn3     = nn.BatchNorm2d(32)
        self.ae_conv4   = nn.Conv2d(32, 32,
                                    kernel_size=3,
                                    bias=False,
                                    padding=1)
        self.ae_bn4     = nn.BatchNorm2d(32)
        self.ae_deconv2 = nn.ConvTranspose2d(32, 16,
                                             bias=False,
                                             stride=2,
                                             padding=1,
                                             kernel_size=4)
        self.ae_bn_d2   = nn.BatchNorm2d(16)
        self.ae_conv5   = nn.Conv2d(16, 16,
                                  kernel_size=3,
                                  bias=False,
                                  padding=1)
        self.ae_bn5     = nn.BatchNorm2d(16)
        self.ae_conv6   = nn.Conv2d(16, 3,
                                    kernel_size=3,
                                    bias=False,
                                    padding=1)

        self.list_actmap = []
        self.list_grad = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        #self.init_relu.register_forward_hook(self._func_forward_)
        #self.layer1.register_forward_hook(self._func_forward_)
        #self.layer2.register_forward_hook(self._func_forward_)
        self.layer3.register_forward_hook(self._func_forward_)

        #self.init_relu.register_backward_hook(self._func_backward_)
        #self.layer1.register_backward_hook(self._func_backward_)
        #self.layer2.register_backward_hook(self._func_backward_)
        self.layer3.register_backward_hook(self._func_backward_)


        self.fc.register_backward_hook(self.__WVN__)

    def __WVN__(self, module, grad_input, grad_output):
        W = module.weight.data
        W_norm = W / torch.norm(W, p=2, dim=1, keepdim=True)
        module.weight.data.copy_(W_norm)
    def _func_forward_(self, module, input, output):
        self.list_actmap.append(output.data)
    def _func_backward_(self, module, grad_input, grad_output):
        self.list_grad.append(grad_output[0].data)
            

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.list_actmap = []
        self.list_grad = []
        # Encoding path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.init_relu(x)    # 32x32

        f1 = self.layer1(x)  # 32x32
        f2 = self.layer2(f1)  # 16x16
        f3 = self.layer3(f2)  # 8x8

        # Decoding path
        dx = self.relu(self.ae_bn1(self.ae_conv1(f3)))
        dx = self.relu(self.ae_bn2(self.ae_conv2(dx)))
        #dx = self.relu(self.ae_bn_d1(self.ae_deconv1(dx))) + f2
        dx = self.relu(self.ae_bn_d1(self.ae_deconv1(dx)))
        dx = self.relu(self.ae_bn3(self.ae_conv3(dx)))
        dx = self.relu(self.ae_bn4(self.ae_conv4(dx)))
        #dx = self.relu(self.ae_bn_d2(self.ae_deconv2(dx))) + f1
        dx = self.relu(self.ae_bn_d2(self.ae_deconv2(dx)))
        dx = self.relu(self.ae_bn5(self.ae_conv5(dx)))
        dx = self.ae_conv6(dx)

        # Classification layers
        x = self.avgpool(f3)
        x = x.view(x.size(0), -1)
        #x = self.bn1d(x)
        feat = x
        #feat = f3.view(x.size(0), -1)
        x = self.fc(x)

        # Feature list : the last element should be recon image
        return x, [feat, f1, f2, f3, dx]


def ae(**kwargs):
    """
    Constructs a ResNet model.
    """
    return AE(**kwargs)
