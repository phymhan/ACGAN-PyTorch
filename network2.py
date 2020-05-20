import torch
import torch.nn as nn
from torch.nn import utils
from resblocks import Block, OptimizedBlock
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import functools
import pdb
from utils import weights_init


class _netG_Res32(nn.Module):
    def __init__(self, ngpu, nz, ny=10, num_classes=10, one_hot=False, ignore_y=False):
        super(_netG_Res32, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ny = ny
        self.one_hot = one_hot
        self.ignore_y = ignore_y

        if self.ignore_y:
            # first linear layer
            self.fc1 = nn.Linear(self.nz, 384)
        else:
            # embed layer for y
            if self.one_hot:
                assert (ny == num_classes)
            else:
                self.embed = nn.Embedding(num_classes, self.ny)
            # first linear layer
            self.fc1 = nn.Linear(self.nz + self.ny, 384)

        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, y=None):
        if self.ignore_y:
            fc1 = self.fc1(z)
        else:
            y_ = torch.zeros(y.size(0), self.ny).cuda().scatter_(1, y.view(-1, 1), 1) if self.one_hot else self.embed(y)
            fc1 = self.fc1(torch.cat([z, y_], dim=1))
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = tconv5
        return output


class _netD_Res32(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, dropout=0.,
                 mi_type_p='ce', mi_type_q='ce', add_eta=True, no_sn_dis=False, use_softmax=False):
        super(_netD_Res32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.add_eta = add_eta
        self.mi_type_p = mi_type_p
        self.mi_type_q = mi_type_q
        self.use_softmax = use_softmax
        self.neg_log_y = np.log(self.num_classes)

        self.block1 = OptimizedBlock(3, num_features, dropout=dropout)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True, dropout=dropout)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True, dropout=dropout)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True, dropout=dropout)
        # dis
        self.linear_dis = utils.spectral_norm(nn.Linear(num_features * 8, 1)) if not no_sn_dis else nn.Linear(num_features * 8, 1)
        # aux for diagnosis
        self.linear_aux = utils.spectral_norm(nn.Linear(num_features * 8, num_classes))
        # P
        if self.mi_type_p == 'ce':
            self.linear_p = utils.spectral_norm(nn.Linear(num_features * 8, num_classes))
        elif self.mi_type_p == 'mine' or self.mi_type_p == 'eta':
            if self.use_softmax:
                self.linear_p = utils.spectral_norm(nn.Linear(num_features * 8, num_classes, bias=False))
            else:
                self.embed_p = utils.spectral_norm(nn.Embedding(num_classes, num_features * 8))
                self.psi_p = utils.spectral_norm(nn.Embedding(num_features * 8, 1))
            self.eta_p = nn.Embedding(num_classes, 1) if self.add_eta else None
        # Q
        if self.mi_type_q == 'ce':
            self.linear_q = utils.spectral_norm(nn.Linear(num_features * 8, num_classes))
        elif self.mi_type_q == 'mine' or self.mi_type_q == 'eta':
            if self.use_softmax:
                self.linear_q = utils.spectral_norm(nn.Linear(num_features * 8, num_classes, bias=False))
            else:
                self.embed_q = utils.spectral_norm(nn.Embedding(num_classes, num_features * 8))
                self.psi_q = utils.spectral_norm(nn.Embedding(num_features * 8, 1))
            self.eta_q = nn.Embedding(num_classes, 1) if self.add_eta else None

    def forward(self, x, y=None, distribution=''):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))

        if y is None:
            realfake = self.linear_dis(h)
            classes = self.linear_aux(h.detach())
            return realfake.squeeze(1), classes

        # if ce, return loglikelihood(y|x); if mine, return t(x, y)
        if distribution == 'P':
            if self.mi_type_p == 'ce':
                t_p = -F.cross_entropy(self.linear_p(h), y, reduction='none').view(-1, 1)
            elif self.mi_type_p == 'mine' or self.mi_type_p == 'eta':
                eta_p = self.eta_p(y) if self.add_eta else 0.
                if self.use_softmax:
                    t_p = -F.cross_entropy(self.linear_p(h), y, reduction='none').view(-1, 1) + eta_p + self.neg_log_y
                else:
                    t_p = torch.sum(self.embed_p(y) * h, dim=1, keepdim=True) + self.psi_p(x) + eta_p + self.neg_log_y
            return t_p.squeeze(1)
        else:
            if self.mi_type_q == 'ce':
                t_q = -F.cross_entropy(self.linear_q(h), y, reduction='none').view(-1, 1)
            elif self.mi_type_q == 'mine' or self.mi_type_p == 'eta':
                eta_q = self.eta_q(y) if self.add_eta else 0.
                if self.use_softmax:
                    t_q = -F.cross_entropy(self.linear_q(h), y, reduction='none').view(-1, 1) + eta_q + self.neg_log_y
                else:
                    t_q = torch.sum(self.embed_q(y) * h, dim=1, keepdim=True) + self.psi_q(x) + eta_q + self.neg_log_y
            return t_q.squeeze(1)

    def log_prob(self, x, y, distribution='P'):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))

        if distribution == 'P':
            if self.mi_type_p == 'ce':
                t_p = -F.cross_entropy(self.linear_p(h), y, reduction='none').view(-1, 1)
            elif self.mi_type_p == 'mine' or self.mi_type_p == 'eta':
                if self.use_softmax:
                    t_p = -F.cross_entropy(self.linear_p(h), y, reduction='none').view(-1, 1)
                else:
                    t_p = torch.sum(self.embed_p(y) * h, dim=1, keepdim=True) + self.psi_p(x)
            return t_p.squeeze(1)
        else:
            if self.mi_type_q == 'ce':
                t_q = -F.cross_entropy(self.linear_q(h), y, reduction='none').view(-1, 1)
            elif self.mi_type_q == 'mine' or self.mi_type_p == 'eta':
                if self.use_softmax:
                    t_q = -F.cross_entropy(self.linear_q(h), y, reduction='none').view(-1, 1)
                else:
                    t_q = torch.sum(self.embed_q(y) * h, dim=1, keepdim=True) + self.psi_q(x)
            return t_q.squeeze(1)

    def get_feature(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        return h
    
    def get_linear_name(self):
        return ['linear_dis', 'linear_p', 'embed_p', 'psi_p', 'eta_p', 'linear_q', 'embed_q', 'psi_q', 'eta_q']

    def get_linear(self):
        params = []
        for param in self.get_linear_name():
            param = getattr(self, param, None)
            if isinstance(param, nn.Sequential):
                param = param[-1]
            params.append(param.weight.data.clone().cpu().numpy())
        return params


# Hinge Loss
def loss_hinge_gen(output):
    loss = torch.mean(F.relu(1. + output))
    return loss

def loss_idt_gen(output):
    loss = torch.mean(output)
    return loss


class _netD2_Res32(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, dropout=0.,
                 mi_type_p='ce', mi_type_q='ce', add_eta=True, no_sn_dis=False, use_softmax=False):
        super(_netD2_Res32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.add_eta = add_eta
        self.mi_type_p = mi_type_p
        self.mi_type_q = mi_type_q
        self.use_softmax = use_softmax
        self.neg_log_y = np.log(self.num_classes)

        self.block1 = OptimizedBlock(3, num_features, dropout=dropout)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True, dropout=dropout)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True, dropout=dropout)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True, dropout=dropout)
        # dis
        self.linear_dis = utils.spectral_norm(nn.Linear(num_features * 8, 1)) if not no_sn_dis else nn.Linear(num_features * 8, 1)
        # aux for diagnosis
        self.linear_aux = utils.spectral_norm(nn.Linear(num_features * 8, num_classes))
        # P
        if self.mi_type_p == 'ce':
            self.linear_p = utils.spectral_norm(nn.Linear(num_features * 8, num_classes))
        elif self.mi_type_p == 'mine' or self.mi_type_p == 'eta':
            if self.use_softmax:
                self.linear_p = utils.spectral_norm(nn.Linear(num_features * 8, num_classes, bias=False))
            else:
                self.embed_p = utils.spectral_norm(nn.Embedding(num_classes, num_features * 8))
                self.psi_p = utils.spectral_norm(nn.Embedding(num_features * 8, 1))
            self.eta_p = nn.Embedding(num_classes, 1) if self.add_eta else None
        # Q
        if self.mi_type_q == 'ce':
            self.linear_q = utils.spectral_norm(nn.Linear(num_features * 8, num_classes))
        elif self.mi_type_q == 'mine' or self.mi_type_q == 'eta':
            if self.use_softmax:
                self.linear_q = utils.spectral_norm(nn.Linear(num_features * 8, num_classes, bias=False))
            else:
                self.embed_q = utils.spectral_norm(nn.Embedding(num_classes, num_features * 8))
                self.psi_q = utils.spectral_norm(nn.Embedding(num_features * 8, 1))
            self.eta_q = nn.Embedding(num_classes, 1) if self.add_eta else None

    def forward(self, x, y=None, distribution=''):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))

        if not distribution:
            realfake = self.linear_dis(h)
            classes = self.linear_aux(h.detach())
            if y is not None:
                real_output = self.get_inner_prod(h, y, 'P')
                fake_output = self.get_inner_prod(h, y, 'Q')
                realfake = realfake + real_output - fake_output
            return realfake.squeeze(1), classes

        # if ce, return loglikelihood(y|x); if mine, return t(x, y)
        if distribution == 'P':
            if self.mi_type_p == 'ce':
                t_p = -F.cross_entropy(self.linear_p(h), y, reduction='none').view(-1, 1)
            elif self.mi_type_p == 'mine' or self.mi_type_p == 'eta':
                eta_p = self.eta_p(y) if self.add_eta else 0.
                if self.use_softmax:
                    t_p = -F.cross_entropy(self.linear_p(h), y, reduction='none').view(-1, 1) + eta_p + self.neg_log_y
                else:
                    t_p = torch.sum(self.embed_p(y) * h, dim=1, keepdim=True) + self.psi_p(x) + eta_p + self.neg_log_y
            return t_p.squeeze(1)
        else:
            if self.mi_type_q == 'ce':
                t_q = -F.cross_entropy(self.linear_q(h), y, reduction='none').view(-1, 1)
            elif self.mi_type_q == 'mine' or self.mi_type_p == 'eta':
                eta_q = self.eta_q(y) if self.add_eta else 0.
                if self.use_softmax:
                    t_q = -F.cross_entropy(self.linear_q(h), y, reduction='none').view(-1, 1) + eta_q + self.neg_log_y
                else:
                    t_q = torch.sum(self.embed_q(y) * h, dim=1, keepdim=True) + self.psi_q(x) + eta_q + self.neg_log_y
            return t_q.squeeze(1)
    
    def get_inner_prod(self, h, y, distribution='P'):
        if distribution == 'P':
            if self.mi_type_p == 'ce':
                t = self.linear_p(h)[range(y.size(0)), y].view(-1, 1)
            elif self.mi_type_p == 'mine' or self.mi_type_p == 'eta':
                if self.use_softmax:
                    t = self.linear_p(h)[range(y.size(0)), y].view(-1, 1)
                else:
                    t = torch.sum(self.embed_p(y) * h, dim=1, keepdim=True)
        else:
            if self.mi_type_q == 'ce':
                t = self.linear_q(h)[range(y.size(0)), y].view(-1, 1)
            elif self.mi_type_q == 'mine' or self.mi_type_q == 'eta':
                if self.use_softmax:
                    t = self.linear_q(h)[range(y.size(0)), y].view(-1, 1)
                else:
                    t = torch.sum(self.embed_q(y) * h, dim=1, keepdim=True)
        return t

    def log_prob(self, x, y, distribution='P'):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))

        if distribution == 'P':
            if self.mi_type_p == 'ce':
                t_p = -F.cross_entropy(self.linear_p(h), y, reduction='none').view(-1, 1)
            elif self.mi_type_p == 'mine' or self.mi_type_p == 'eta':
                if self.use_softmax:
                    t_p = -F.cross_entropy(self.linear_p(h), y, reduction='none').view(-1, 1)
                else:
                    t_p = torch.sum(self.embed_p(y) * h, dim=1, keepdim=True) + self.psi_p(x)
            return t_p.squeeze(1)
        else:
            if self.mi_type_q == 'ce':
                t_q = -F.cross_entropy(self.linear_q(h), y, reduction='none').view(-1, 1)
            elif self.mi_type_q == 'mine' or self.mi_type_p == 'eta':
                if self.use_softmax:
                    t_q = -F.cross_entropy(self.linear_q(h), y, reduction='none').view(-1, 1)
                else:
                    t_q = torch.sum(self.embed_q(y) * h, dim=1, keepdim=True) + self.psi_q(x)
            return t_q.squeeze(1)

    def get_feature(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        return h
    
    def get_linear_name(self):
        return ['linear_dis', 'linear_p', 'embed_p', 'psi_p', 'eta_p', 'linear_q', 'embed_q', 'psi_q', 'eta_q']

    def get_linear(self):
        params = []
        for param in self.get_linear_name():
            param = getattr(self, param, None)
            if isinstance(param, nn.Sequential):
                param = param[-1]
            params.append(param.weight.data.clone().cpu().numpy())
        return params
