# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization import *
from functools import partial


def get_act(config):
    if config.model.nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == 'swish':
        def swish(x):
            return x * torch.sigmoid(x)
        return swish
    else:
        raise NotImplementedError('activation function does not exist!')

def spectral_norm(layer, n_iters=1):
    return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)

def conv1x1(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "1x1 convolution"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def conv3x3(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv


def stride_conv3x3(in_planes, out_planes, kernel_size, bias=True, spec_norm=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2,
                     padding=kernel_size // 2, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True, spec_norm=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv

class CRPBlock(nn.Module):
    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True, spec_norm=False):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))
        self.n_stages = n_stages
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class CondCRPBlock(nn.Module):
    def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU(), spec_norm=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.normalizer = normalizer
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_classes, bias=True))
            self.convs.append(conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))

        self.n_stages = n_stages
        self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.maxpool(path)
            path = self.convs[i](path)

            x = path + x
        return x


class RCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU(), spec_norm=False):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1), conv3x3(features, features, stride=1, bias=False,
                                                                         spec_norm=spec_norm))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

            x += residual
        return x


class CondRCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU(), spec_norm=False):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_norm'.format(i + 1, j + 1), normalizer(features, num_classes, bias=True))
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1),
                        conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.normalizer = normalizer

    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

            x += residual
        return x


class MSFBlock(nn.Module):
    def __init__(self, in_planes, features, spec_norm=False):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm))

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class CondMSFBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer, spec_norm=False):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features
        self.normalizer = normalizer

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class RefineBlock(nn.Module):
    def __init__(self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True, spec_norm=False):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                RCUBlock(in_planes[i], 2, 2, act, spec_norm=spec_norm)
            )

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act, spec_norm=spec_norm)

        if not start:
            self.msf = MSFBlock(in_planes, features, spec_norm=spec_norm)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool, spec_norm=spec_norm)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h



class CondRefineBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer, act=nn.ReLU(), start=False, end=False, spec_norm=False):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act, spec_norm=spec_norm)
            )

        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act, spec_norm=spec_norm)

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer, spec_norm=spec_norm)

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act, spec_norm=spec_norm)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False, spec_norm=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            if spec_norm:
                conv = spectral_norm(conv)
            self.conv = conv
        else:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            if spec_norm:
                conv = spectral_norm(conv)

            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                conv
            )
        self.avg = nn.AvgPool2d( kernel_size=2 )

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.avg( output )
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, spec_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        if spec_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, inputs):
        output = inputs
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                      output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return self.conv(output)


class UpsampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, spec_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        if spec_norm:
            self.conv = spectral_norm(self.conv)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, resample=None, act=nn.ELU(),
                 normalization=ConditionalBatchNorm2d, adjust_padding=False, dilation=None, spec_norm=False):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding, spec_norm=spec_norm)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding, spec_norm=spec_norm)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_classes)


    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
                 normalization=nn.BatchNorm2d, adjust_padding=False, dilation=None, spec_norm=False):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding, spec_norm=spec_norm)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding, spec_norm=spec_norm)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(conv1x1, spec_norm=spec_norm)
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)


    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output
