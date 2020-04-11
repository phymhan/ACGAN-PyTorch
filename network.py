import torch
import torch.nn as nn
from torch.nn import utils
from resblocks import Block, OptimizedBlock
from torch.nn import init
import torch.nn.functional as F
import functools
import pdb


# Identity mapping
class IdentityMapping(nn.Module):
    def __init__(self, *args):
        super(IdentityMapping, self).__init__()

    def forward(self, x):
        return x


class _netG(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(nz, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv6, tconv5, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            tconv5 = self.tconv6(tconv5)
            output = tconv5
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, num_classes=10, tac=False):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.tac = tac

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(13*13*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(13*13*512, num_classes)
        # twin aux-classifier fc
        self.fc_tac = nn.Linear(13 * 13 * 512, num_classes) if self.tac else None
        # softmax and sigmoid
        # self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 13*13*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
            fc_tac = nn.parallel.data_parallel(self.fc_tac, flat6, range(self.ngpu)) if self.tac else None
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 13*13*512)
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)
            fc_tac = self.fc_tac(flat6) if self.tac else None
        classes = fc_aux
        realfake = fc_dis.squeeze(1)
        if self.tac:
            classes_twin = fc_tac
            return realfake, classes, classes_twin
        else:
            return realfake, classes


class _netT(nn.Module):
    def __init__(self, ngpu):
        super(_netT, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # mine fc
        self.fc_mine = nn.Linear(13 * 13 * 512, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=2e-1)
        self.ma_et = None

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 13*13*512)
            fc_mine = nn.parallel.data_parallel(self.fc_mine, flat6, range(self.ngpu))
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 13*13*512)
            fc_mine = self.fc_mine(flat6)
        return self.lrelu(fc_mine)


class _netG_CIFAR10(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG_CIFAR10, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(nz, 384)
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

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            output = tconv5
        return output


class _netD_CIFAR10(nn.Module):
    def __init__(self, ngpu, num_classes=10, tac=False, dropout=0.):
        super(_netD_CIFAR10, self).__init__()
        self.ngpu = ngpu
        self.tac = tac

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # twin aux-classifier fc
        self.fc_tac = nn.Linear(4*4*512, num_classes) if self.tac else None
        # softmax and sigmoid
        # self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
            fc_tac = nn.parallel.data_parallel(self.fc_tac, flat6, range(self.ngpu)) if self.tac else None
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)
            fc_tac = self.fc_tac(flat6) if self.tac else None
        classes = fc_aux
        realfake = fc_dis.squeeze(1)
        if self.tac:
            classes_twin = fc_tac
            return realfake, classes, classes_twin
        else:
            return realfake, classes


class _netT_concat_CIFAR10(nn.Module):
    def __init__(self, ngpu):
        super(_netT_concat_CIFAR10, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3+10, 16, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # mine fc
        self.fc_mine = nn.Linear(4*4*512, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=2e-1)
        self.ma_et = None

    def forward(self, x, z):
        # z is y
        z_onehot = torch.zeros(x.size(0), 10, 1, 1).cuda().scatter_(1, z.view(x.size(0), 1, 1, 1), 1)
        xz = torch.cat((x, z_onehot.expand(x.size(0), 10, x.size(2), x.size(3))), 1)
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, xz, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 4*4*512)
            fc_mine = nn.parallel.data_parallel(self.fc_mine, flat6, range(self.ngpu))
        else:
            conv1 = self.conv1(xz)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 4*4*512)
            fc_mine = self.fc_mine(flat6)
        # return self.lrelu(fc_mine)
        return fc_mine


# ref: https://github.com/crcrpar/pytorch.sngan_projection/blob/master/models/discriminators/snresnet.py
class _netDT_CIFAR10(nn.Module):
    def __init__(self, ngpu, num_classes=10):
        super(_netDT_CIFAR10, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        # Embedding for label
        self.l_y = utils.spectral_norm(nn.Embedding(num_classes, 512))  # TODO: add spectral norm

        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # mine fc
        self.fc_mine = nn.Linear(4*4*512, 1)
        # softmax and sigmoid
        # self.sigmoid = nn.Sigmoid()
        self.ma_et = None

    def forward(self, x, y=None):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 4*4*512)

        if y is not None:
            # y = torch.zeros(x.size(0), self.num_classes).cuda().scatter_(1, y.view(x.size(0), 1), 1)
            # global pooling
            feat = torch.sum(conv6, dim=(2, 3))
            Vy = self.l_y(y)
            inner_prod = torch.sum(Vy * feat, dim=1, keepdim=True)
            T = inner_prod + self.fc_mine(flat6)
            return T
        else:
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)
            classes = fc_aux
            realfake = fc_dis.squeeze(1)
            return realfake, classes


class _netDT_SNResProj32(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, use_cy=False, dropout=0.,
                 sn_emb_l=True, sn_emb_c=True, init_zero=False, softmax=False):
        super(_netDT_SNResProj32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.use_cy = use_cy
        self.init_zero = init_zero
        self.softmax = softmax

        self.block1 = OptimizedBlock(3, num_features, dropout=dropout)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True, dropout=dropout)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True, dropout=dropout)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True, dropout=dropout)
        if num_classes > 0:
            if softmax:
                self.fc_t = utils.spectral_norm(nn.Linear(num_features * 8, num_classes))
            else:
                self.l5 = utils.spectral_norm(nn.Linear(num_features * 8, 1))
                self.l_y = utils.spectral_norm(nn.Embedding(num_classes, num_features * 8)) if sn_emb_l else nn.Embedding(num_classes, num_features * 8)
            if use_cy:
                self.c_y = utils.spectral_norm(nn.Embedding(num_classes, 1)) if sn_emb_c else nn.Embedding(num_classes, 1)
        self.ma_et = None

        # discriminator fc
        self.fc_dis = utils.spectral_norm(nn.Linear(num_features * 8, 1))
        # aux-classifier fc
        self.fc_aux = utils.spectral_norm(nn.Linear(num_features * 8, num_classes))

        self._initialize()

    def _initialize(self):
        optional_l5 = getattr(self, 'l5', None)
        if optional_l5 is not None:
            init.xavier_uniform_(optional_l5.weight.data)
        optional_fc_t = getattr(self, 'fc_t', None)
        if optional_fc_t is not None:
            init.xavier_uniform_(optional_fc_t.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)
        optional_c_y = getattr(self, 'c_y', None)
        if optional_c_y is not None:
            if self.init_zero:
                optional_c_y.weight.data.fill_(0)
            else:
                init.xavier_uniform_(optional_c_y.weight.data)

    def forward(self, x, y=None):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))

        if y is not None:
            # netT(x, y)
            cy = self.c_y(y) if self.use_cy else 0.0
            if self.softmax:
                logprob = torch.log(torch.softmax(self.fc_t(h), dim=1))
                output = logprob[range(y.size(0)), y] + cy
            else:
                output = self.l5(h)
                output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True) + cy
            return output
        else:
            # netD(x)
            realfake = self.fc_dis(h).squeeze(1)
            classes = self.fc_aux(h)
            return realfake, classes


class _netDT2_SNResProj32(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, use_cy=True, ac=False, tac=False, dropout=0.,
                 sn_emb_l=True, sn_emb_c=True, init_zero=False):
        super(_netDT2_SNResProj32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.use_cy = use_cy
        self.ac = ac
        self.tac = tac
        self.init_zero = init_zero

        self.block1 = OptimizedBlock(3, num_features, dropout=dropout)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True, dropout=dropout)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True, dropout=dropout)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True, dropout=dropout)
        self.l5_P = utils.spectral_norm(nn.Linear(num_features * 8, 1))
        self.l5_Q = utils.spectral_norm(nn.Linear(num_features * 8, 1))
        if num_classes > 0:
            self.l_y_P = utils.spectral_norm(nn.Embedding(num_classes, num_features * 8)) if sn_emb_l else nn.Embedding(num_classes, num_features * 8)
            self.l_y_Q = utils.spectral_norm(nn.Embedding(num_classes, num_features * 8)) if sn_emb_l else nn.Embedding(num_classes, num_features * 8)
            if use_cy:
                self.c_y_P = utils.spectral_norm(nn.Embedding(num_classes, 1)) if sn_emb_c else nn.Embedding(num_classes, 1)
                self.c_y_Q = utils.spectral_norm(nn.Embedding(num_classes, 1)) if sn_emb_c else nn.Embedding(num_classes, 1)
        self.ma_et_P = None
        self.ma_et_Q = None

        # discriminator fc
        self.fc_dis = utils.spectral_norm(nn.Linear(num_features * 8, 1))
        # aux-classifier fc
        self.fc_aux = utils.spectral_norm(nn.Linear(num_features * 8, num_classes)) if ac else None
        # twin aux-classifier fc
        self.tac_aux = utils.spectral_norm(nn.Linear(num_features * 8, num_classes)) if tac else None

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l5_P.weight.data)
        init.xavier_uniform_(self.l5_Q.weight.data)
        optional_l_y_P = getattr(self, 'l_y_P', None)
        if optional_l_y_P is not None:
            init.xavier_uniform_(optional_l_y_P.weight.data)
        optional_l_y_Q = getattr(self, 'l_y_Q', None)
        if optional_l_y_Q is not None:
            init.xavier_uniform_(optional_l_y_Q.weight.data)
        optional_c_y_P = getattr(self, 'c_y_P', None)
        if optional_c_y_P is not None:
            if self.init_zero:
                optional_c_y_P.weight.data.fill_(0.)
            else:
                init.xavier_uniform_(optional_c_y_P.weight.data)
        optional_c_y_Q = getattr(self, 'c_y_Q', None)
        if optional_c_y_Q is not None:
            if self.init_zero:
                optional_c_y_Q.weight.data.fill_(0.)
            else:
                init.xavier_uniform_(optional_c_y_Q.weight.data)

    def forward(self, x, y=None, distribution='P'):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))

        if y is not None:
            if distribution == 'P':
                output = self.l5_P(h)
                cy = self.c_y_P(y) if self.use_cy else 0.0
                output += torch.sum(self.l_y_P(y) * h, dim=1, keepdim=True) + cy
            elif distribution == 'Q':
                output = self.l5_Q(h)
                cy = self.c_y_Q(y) if self.use_cy else 0.0
                output += torch.sum(self.l_y_Q(y) * h, dim=1, keepdim=True) + cy
            else:
                raise RuntimeError
            return output.squeeze(1)
        else:
            realfake = self.fc_dis(h).squeeze(1)
            classes = self.fc_aux(h) if self.ac else None
            if self.tac:
                classes_twin = self.tac_aux(h)
                return realfake, classes, classes_twin
            else:
                return realfake, classes

    def log_prob(self, x, y, distribution='P'):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))

        if distribution == 'P':
            output = self.l5_P(h)
            output += torch.sum(self.l_y_P(y) * h, dim=1, keepdim=True)
        elif distribution == 'Q':
            output = self.l5_Q(h)
            output += torch.sum(self.l_y_Q(y) * h, dim=1, keepdim=True)
        else:
            raise RuntimeError
        return output.squeeze(1)


class _netD_SNRes32(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, tac=False, dropout=0.):
        super(_netD_SNRes32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.tac = tac

        self.block1 = OptimizedBlock(3, num_features, dropout=dropout)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True, dropout=dropout)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True, dropout=dropout)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True, dropout=dropout)

        # discriminator fc
        self.fc_dis = utils.spectral_norm(nn.Linear(1 * 1 * 512, 1))
        # aux-classifier fc
        self.fc_aux = utils.spectral_norm(nn.Linear(1 * 1 * 512, num_classes))
        # twin aux-classifier fc
        self.tac_aux = utils.spectral_norm(nn.Linear(1 * 1 * 512, num_classes))

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))

        realfake = self.fc_dis(h).squeeze(1)
        classes = self.fc_aux(h)
        if self.tac:
            classes_twin = self.tac_aux(h)
            return realfake, classes, classes_twin
        else:
            return realfake, classes


# borrowed from https://github.com/crcrpar/pytorch.sngan_projection/blob/master/models/discriminators/snresnet.py
class SNResNetProjectionDiscriminator64(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, use_cy=False, sn_emb_l=True, sn_emb_c=True):
        super(SNResNetProjectionDiscriminator64, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.use_cy = use_cy

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16,
                            activation=activation, downsample=True)
        self.l6 = utils.spectral_norm(nn.Linear(num_features * 16, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, num_features * 16)) if sn_emb_l else nn.Embedding(num_classes, num_features * 16)
            self.c_y = utils.spectral_norm(nn.Embedding(num_classes, 1)) if sn_emb_c else nn.Embedding(num_classes, 1)
        self.ma_et = None
        self.ma_et_P = None
        self.ma_et_Q = None
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None, distribution='P'):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        if y is not None:
            cy = self.c_y(y) if self.use_cy else 0.0
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True) + cy
        return output.squeeze(1)

    def log_prob(self, x, y, distribution='P'):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output.squeeze(1)


class SNResNetProjectionDiscriminator32(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, use_cy=False, sn_emb_l=True, sn_emb_c=True):
        super(SNResNetProjectionDiscriminator32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.use_cy = use_cy

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.l5 = utils.spectral_norm(nn.Linear(num_features * 8, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, num_features * 8)) if sn_emb_l else nn.Embedding(num_classes, num_features * 8)
            self.c_y = utils.spectral_norm(nn.Embedding(num_classes, 1)) if sn_emb_c else nn.Embedding(num_classes, 1)
        self.ma_et = None
        self.ma_et_P = None
        self.ma_et_Q = None
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l5.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None, distribution='P'):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)
        if y is not None:
            cy = self.c_y(y) if self.use_cy else 0.0
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True) + cy
        return output.squeeze(1)

    def log_prob(self, x, y, distribution='P'):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output.squeeze(1)


## Latent
class EmbeddingNet(nn.Module):
    def __init__(self, nz=64, K=0):
        super(EmbeddingNet, self).__init__()
        self.nz = nz
        self.K = K
        self.A = utils.spectral_norm(nn.Embedding(K, nz))
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.A.weight.data)

    def forward(self, k):
        return self.A(k)


class ReconstructorConcat(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(ReconstructorConcat, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3 + 3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)

        # discriminator fc
        self.fc_class = utils.spectral_norm(nn.Linear(1 * 1 * 512, num_classes))
        # aux-classifier fc
        self.fc_shift = utils.spectral_norm(nn.Linear(1 * 1 * 512, 1))

    def forward(self, x1, x2):
        h = torch.cat((x1, x2), dim=1)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))

        classes = self.fc_class(h)
        shifted = self.fc_shift(h)
        return classes, shifted


class ReconstructorSiamese(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(ReconstructorSiamese, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)

        # discriminator fc
        self.fc_class = utils.spectral_norm(nn.Linear(1 * 1 * 512 * 2, num_classes))
        # aux-classifier fc
        self.fc_shift = utils.spectral_norm(nn.Linear(1 * 1 * 512, 1))

    def forward_once(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        return h

    def forward(self, x1, x2):
        h1 = self.forward_once(x1)
        h2 = self.forward_once(x2)

        classes = self.fc_class(torch.cat((h1, h2), dim=1))
        shifted = self.fc_shift(h2) - self.fc_shift(h1)
        return classes, shifted
