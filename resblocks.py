import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils
import functools
import network


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, dropout=0.):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample
        if dropout > 0:
            drop_layer = functools.partial(nn.Dropout2d, p=dropout)
        else:
            drop_layer = network.IdentityMapping

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        self.drop1 = drop_layer()
        self.drop2 = drop_layer()
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
            self.drop_sc = drop_layer()

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.drop_sc(self.c_sc(x))
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        # conv(activation(x))?
        h = self.drop1(self.c1(self.activation(x)))
        h = self.drop2(self.c2(self.activation(h)))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu, dropout=0.):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        if dropout > 0:
            drop_layer = functools.partial(nn.Dropout2d, p=dropout)
        else:
            drop_layer = network.IdentityMapping

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        self.drop1 = drop_layer()
        self.drop2 = drop_layer()
        self.drop_sc = drop_layer()

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.drop_sc(self.c_sc(F.avg_pool2d(x, 2)))

    def residual(self, x):
        h = self.activation(self.drop1(self.c1(x)))
        return F.avg_pool2d(self.drop2(self.c2(h)), 2)
