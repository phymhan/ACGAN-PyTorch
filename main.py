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
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10, _netD_SNRes32, SNResNetProjectionDiscriminator32
from folder import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from inception import prepare_inception_metrics
import torch.nn.functional as F
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ny', type=int, default=0, help='size of the latent embedding vector for y')
parser.add_argument('--use_onehot_embed', action='store_true', help='use onehot embedding in G?')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--loss_type', type=str, default='ac', help='[ac | tac | cgan]')
parser.add_argument('--visualize_class_label', type=int, default=-1, help='if < 0, random int')
parser.add_argument('--lambda_tac', type=float, default=1.0)
parser.add_argument('--download_dset', action='store_true')
parser.add_argument('--num_inception_images', type=int, default=10000)
parser.add_argument('--netD_model', type=str, default='basic', help='[basic | snres32]')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--bnn_dropout', type=float, default=0.)
# parser.add_argument('--shuffle_label', type=str, default='uniform', help='[uniform | shuffle | same]')
parser.add_argument('--label_rotation', action='store_true')
parser.add_argument('--disable_cudnn_benchmark', action='store_true')

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

if opt.disable_cudnn_benchmark:
    cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
else:
    cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

writer = SummaryWriter(log_dir=opt.outf)

# datase t
if opt.dataset == 'imagenet':
    # folder dataset
    opt.imageSize = 128
    dataset = ImageFolder(
        root=opt.dataroot,
        transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        classes_idx=(10, 20)
    )
elif opt.dataset == 'cifar10':
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
nz = int(opt.nz)
ny = int(opt.num_classes) if opt.ny == 0 else int(opt.ny)  # embedding dim same as onehot embedding by default
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3
tac = opt.loss_type == 'tac'

# Define the generator and initialize the weights
if opt.dataset == 'imagenet':
    netG = _netG(ngpu, nz)
elif opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
    netG = _netG_CIFAR10(ngpu, nz, ny, num_classes, one_hot=opt.use_onehot_embed)
elif opt.dataset == 'mnist':
    netG = _netG_CIFAR10(ngpu, nz, ny, num_classes, one_hot=opt.use_onehot_embed)
else:
    raise NotImplementedError
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Define the discriminator and initialize the weights
if opt.dataset == 'imagenet':
    netD = _netD(ngpu, num_classes, tac=opt.loss_type == 'tac')
elif opt.dataset == 'mnist' or opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
    if opt.loss_type == 'cgan':
        if opt.netD_model == 'snres32':
            netD = SNResNetProjectionDiscriminator32(opt.ndf, opt.num_classes, use_cy=False)
        else:
            raise NotImplementedError
    else:
        if opt.netD_model == 'snres32':
            netD = _netD_SNRes32(opt.ndf, opt.num_classes, tac=opt.loss_type == 'tac', dropout=opt.bnn_dropout)
        elif opt.netD_model == 'basic':
            netD = _netD_CIFAR10(ngpu, num_classes, tac=opt.loss_type == 'tac')
            netD.apply(weights_init)
        else:
            raise NotImplementedError
else:
    raise NotImplementedError
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# loss functions
dis_criterion = nn.BCEWithLogitsLoss()
aux_criterion = nn.CrossEntropyLoss()

# tensor placeholders
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.randn(opt.batchSize, nz, requires_grad=False)
eval_noise = torch.randn(opt.batchSize, nz, requires_grad=False)
fake_label = torch.LongTensor(opt.batchSize)
dis_label = torch.FloatTensor(opt.batchSize)
real_label_const = 1
fake_label_const = 0

# noise for evaluation
eval_label_const = 0
eval_label = torch.LongTensor(opt.batchSize).random_(0, num_classes)
if opt.visualize_class_label >= 0:
    eval_label_const = opt.visualize_class_label % opt.num_classes
    eval_label.data.fill_(eval_label_const)

# if using cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    input, dis_label, fake_label, eval_label = input.cuda(), dis_label.cuda(), fake_label.cuda(), eval_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

dset_name = os.path.split(opt.dataroot)[-1]
datafile = os.path.join(opt.dataroot, '..', f'{dset_name}_stats', dset_name)
sampler = ImageSampler(netG, opt)
get_metrics = prepare_inception_metrics(dataloader, datafile, False, opt.num_inception_images, no_is=False)

losses_D = []
losses_G = []
losses_A = []
losses_F = []
losses_I_mean = []
losses_I_std = []
for epoch in range(opt.niter):
    avg_loss_D = AverageMeter()
    avg_loss_G = AverageMeter()
    avg_loss_A = AverageMeter()
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, label = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu, label = real_cpu.cuda(), label.cuda()
        with torch.no_grad():
            input.resize_as_(real_cpu).copy_(real_cpu)
            dis_label.resize_(batch_size).fill_(real_label_const)
        if opt.loss_type == 'cgan':
            dis_output = netD(input, label)
        elif opt.loss_type == 'tac':
            dis_output, aux_output, _ = netD(input)
        elif opt.loss_type == 'ac':
            dis_output, aux_output = netD(input)
        else:
            raise RuntimeError
        dis_errD_real = dis_criterion(dis_output, dis_label)
        if opt.loss_type == 'cgan':
            aux_errD_real = 0.
        else:
            aux_errD_real = aux_criterion(aux_output, label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = torch.sigmoid(dis_output).data.mean()

        # compute the current classification accuracy
        if opt.loss_type == 'cgan':
            accuracy = 1. / opt.num_classes
        else:
            accuracy = compute_acc(aux_output, label)

        # get fake
        fake_label.resize_(batch_size).random_(0, num_classes)
        noise.resize_(batch_size, nz).normal_(0, 1)
        fake = netG(noise, fake_label)

        # train with fake
        dis_label.resize_(batch_size).fill_(fake_label_const)
        if opt.loss_type == 'cgan':
            dis_output = netD(fake.detach(), fake_label)
            tac_errD_fake = 0.
        elif opt.loss_type == 'tac':
            dis_output, aux_output, tac_output = netD(fake.detach())
            # tac on fake
            tac_errD_fake = aux_criterion(tac_output, fake_label)
        elif opt.loss_type == 'ac':
            dis_output, aux_output = netD(fake.detach())
            tac_errD_fake = 0.
        else:
            raise RuntimeError

        dis_errD_fake = dis_criterion(dis_output, dis_label)
        if opt.loss_type == 'cgan':
            aux_errD_fake = 0.
        else:
            aux_errD_fake = aux_criterion(aux_output, fake_label)
        errD_fake = dis_errD_fake + aux_errD_fake + tac_errD_fake
        errD_fake.backward()
        D_G_z1 = torch.sigmoid(dis_output).data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label_const)  # fake labels are real for generator cost
        if opt.loss_type == 'cgan':
            dis_output = netD(fake, fake_label)
            tac_errG = 0.
        elif opt.loss_type == 'tac':
            dis_output, aux_output, tac_output = netD(fake)
            tac_errG = aux_criterion(tac_output, fake_label)
        elif opt.loss_type == 'ac':
            dis_output, aux_output = netD(fake)
            tac_errG = 0.
        else:
            raise RuntimeError
        dis_errG = dis_criterion(dis_output, dis_label)
        if opt.loss_type == 'cgan':
            aux_errG = 0.
        else:
            aux_errG = aux_criterion(aux_output, fake_label)
        errG = dis_errG + aux_errG - opt.lambda_tac * tac_errG
        errG.backward()
        D_G_z2 = torch.sigmoid(dis_output).data.mean()
        optimizerG.step()

        # compute the average loss
        avg_loss_G.update(errG.item(), batch_size)
        avg_loss_D.update(errD.item(), batch_size)
        avg_loss_A.update(accuracy, batch_size)

        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), avg_loss_D.avg,
                 errG.item(), avg_loss_G.avg,
                 D_x, D_G_z1, D_G_z2,
                 accuracy, avg_loss_A.avg))
        if i % 100 == 0:
            vutils.save_image(
                real_cpu, '%s/real_samples.png' % opt.outf)
            # print('Label for eval = {}'.format(eval_label))
            fake = netG(eval_noise, eval_label)
            vutils.save_image(
                fake.data,
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)
            )

    # update eval_label
    if opt.visualize_class_label >= 0 and opt.label_rotation:
        eval_label_const = (eval_label_const + 1) % num_classes
        eval_label.data.fill_(eval_label_const)

    # compute metrics
    is_mean, is_std, fid = get_metrics(sampler, num_inception_images=opt.num_inception_images, num_splits=10,
                                       prints=True, use_torch=False)
    writer.add_scalar('Loss/G', avg_loss_G.avg, epoch)
    writer.add_scalar('Loss/D', avg_loss_D.avg, epoch)
    writer.add_scalar('Metric/Aux', avg_loss_A.avg, epoch)
    writer.add_scalar('Metric/FID', fid, epoch)
    writer.add_scalar('Metric/IS', is_mean, epoch)
    losses_G.append(avg_loss_G.avg)
    losses_D.append(avg_loss_D.avg)
    losses_A.append(avg_loss_A.avg)
    losses_F.append(fid)
    losses_I_mean.append(is_mean)
    losses_I_std.append(is_std)

    # do checkpointing
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    np.save(f'{opt.outf}/losses_G.npy', np.array(losses_G))
    np.save(f'{opt.outf}/losses_D.npy', np.array(losses_D))
    np.save(f'{opt.outf}/losses_A.npy', np.array(losses_A))
    np.save(f'{opt.outf}/losses_F.npy', np.array(losses_F))
    np.save(f'{opt.outf}/losses_I_mean.npy', np.array(losses_I_mean))
    np.save(f'{opt.outf}/losses_I_std.npy', np.array(losses_I_std))
