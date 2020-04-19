"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import weights_init, compute_acc, AverageMeter, ImageSampler, print_options, set_onehot
from network import _netG, _netD, _netT, _netD_CIFAR10, _netD_SNRes32, _netG_CIFAR10, _netT_concat_CIFAR10, _netDT_CIFAR10
from network import SNResNetProjectionDiscriminator64, SNResNetProjectionDiscriminator32, _netDT_SNResProj32
from folder import ImageFolder
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from inception import prepare_inception_metrics, prepare_data_statistics
import torch.nn.functional as F
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ntf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netT', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--ma_rate', type=float, default=0.001)
parser.add_argument('--download_dset', action='store_true')
parser.add_argument('--use_cy', action='store_true')
parser.add_argument('--no_sn_emb_l', action='store_true')
parser.add_argument('--no_sn_emb_c', action='store_true')
parser.add_argument('--emb_init_zero', action='store_true')
parser.add_argument('--softmax_T', action='store_true')
parser.add_argument('--netD_model', type=str, default='basic', help='[basic | proj32]')
parser.add_argument('--netT_model', type=str, default='concat', help='[concat | proj32 | proj64]')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--bnn_dropout', type=float, default=0.)
parser.add_argument('--weighted_mine_loss', action='store_true', default=False)
parser.add_argument('--label_rotation', action='store_true')
parser.add_argument('--eps', type=float, default=0., help='eps added in log')

opt = parser.parse_args()
print_options(parser, opt)

# specify the gpu id if using only 1 gpu
# if opt.ngpu == 1:
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

writer = SummaryWriter(log_dir=opt.outf)

# datase
if opt.dataset == 'cifar10':
    opt.imageSize = 32
    dataset = dset.CIFAR10(
        root=opt.dataroot, download=True,
        transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
elif opt.dataset == 'cifar100':
    opt.imageSize = 32
    dataset = dset.CIFAR100(
        root=opt.dataroot, download=True,
        transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
elif opt.dataset == 'mnist':
    opt.imageSize = 32
    dataset = dset.MNIST(
        root=opt.dataroot, download=True,
        transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
else:
    raise NotImplementedError("No such dataset {}".format(opt.dataset))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# some hyper parameters
ngpu = int(opt.ngpu)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3

# Define the discriminator and initialize the weights
if opt.dataset == 'mnist' or opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
    if opt.netD_model == 'proj32':
        netD = _netDT_SNResProj32(opt.ndf, opt.num_classes, use_cy=opt.use_cy, dropout=opt.bnn_dropout,
                                  sn_emb_l=not opt.no_sn_emb_l, sn_emb_c=not opt.no_sn_emb_c,
                                  init_zero=opt.emb_init_zero, softmax=opt.softmax_T)
    else:
        raise NotImplementedError
else:
    raise NotImplementedError

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# tensor placeholders
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

# if using cuda
if opt.cuda:
    netD.cuda()
    input = input.cuda()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

dset_name = os.path.split(opt.dataroot)[-1]
datafile = os.path.join(opt.dataroot, '..', f'{dset_name}_stats', dset_name)

losses_M = []

for epoch in range(opt.niter):
    avg_loss_M = AverageMeter()
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu, label = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu, label = real_cpu.cuda(), label.cuda()
        with torch.no_grad():
            input.resize_as_(real_cpu).copy_(real_cpu)

        # train with real
        y = label
        y_bar = y[torch.randperm(batch_size), ...]
        et = torch.mean(torch.exp(netD(input, y_bar)))
        if netD.ma_et is None:
            netD.ma_et = et.detach().item()
        netD.ma_et += opt.ma_rate * (et.detach().item() - netD.ma_et)
        mi = torch.mean(netD(input, y, 'P')) - torch.log(et + 1e-8) * et.detach() / netD.ma_et
        (-mi).backward()
        optimizerD.step()

        # compute the average loss
        avg_loss_M.update(mi.item(), batch_size)

        print('[%d/%d][%d/%d] MI: %.4f (%.4f)' % (epoch, opt.niter, i, len(dataloader), mi.item(), avg_loss_M.avg))

    writer.add_scalar('Metric/MI', avg_loss_M.avg, epoch)
    losses_M.append(avg_loss_M.avg)

    # do checkpointing
    if epoch % 10 == 0:
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    np.save(f'{opt.outf}/losses_M.npy', np.array(losses_M))
