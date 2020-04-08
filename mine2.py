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
from utils import weights_init, compute_acc, AverageMeter, ImageSampler, print_options
from network import _netG, _netD, _netT, _netD_CIFAR10, _netD_SNRes32, _netG_CIFAR10, _netT_concat_CIFAR10, _netDT_CIFAR10
from network import SNResNetProjectionDiscriminator64, SNResNetProjectionDiscriminator32, _netDT2_SNResProj32
from folder import ImageFolder
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from inception import prepare_inception_metrics, prepare_data_statistics
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ntf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netT', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--loss_type', type=str, default='mine', help='[ac | tac | mine]')
parser.add_argument('--visualize_class_label', type=int, default=-1, help='if < 0, random int')
parser.add_argument('--ma_rate', type=float, default=0.001)
parser.add_argument('--lambda_y_grad', type=float, default=1.)
parser.add_argument('--lambda_y', type=float, default=1.)
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--adaptive_grad', type=str, default='dc', help='[d | c | dc]')
parser.add_argument('--n_update_mine', type=int, default=1, help='how many updates on mine in each iteration')
parser.add_argument('--download_dset', action='store_true')
parser.add_argument('--num_inception_images', type=int, default=10000)
parser.add_argument('--use_shared_T', action='store_true')
parser.add_argument('--use_cy', action='store_true')
parser.add_argument('--netD_model', type=str, default='basic', help='[basic | proj32]')
parser.add_argument('--netT_model', type=str, default='concat', help='[concat | proj32 | proj64]')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--bnn_dropout', type=float, default=0.)
parser.add_argument('--f_div', type=str, default='revkl', help='[kl | revkl | pearson | squared | jsd | gan]')
parser.add_argument('--shuffle_label', type=str, default='uniform', help='[uniform | shuffle | same]')

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

# datase t
if opt.dataset == 'imagenet':
    # folder dataset
    opt.imageSize = 128
    # dataset = ImageFolder(
    #     root=opt.dataroot,
    #     transform=transforms.Compose([
    #         transforms.Scale(opt.imageSize),
    #         transforms.CenterCrop(opt.imageSize),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]),
    #     classes_idx=(10, 20)
    # )
    dataset = dset.ImageNet(
        root=opt.dataroot, download=opt.download_dset,
        transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
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
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3

# Define the generator and initialize the weights
if opt.dataset == 'imagenet':
    netG = _netG(ngpu, nz)
elif opt.dataset == 'cifar10':
    netG = _netG_CIFAR10(ngpu, nz)
elif opt.dataset == 'mnist':
    netG = _netG_CIFAR10(ngpu, nz)
else:
    raise NotImplementedError
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Define the discriminator and initialize the weights
if opt.dataset == 'imagenet':
    netD = _netD(ngpu, num_classes)
elif opt.dataset == 'mnist' or opt.dataset == 'cifar10':
    if opt.use_shared_T:
        if opt.netD_model == 'proj32':
            netD = _netDT2_SNResProj32(opt.ndf, opt.num_classes, use_cy=opt.use_cy, dropout=opt.bnn_dropout)
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

# Define the statistics network and initialize the weights
if opt.dataset == 'imagenet':
    netTP = _netT(ngpu)
    netTQ = _netT(ngpu)
else:
    if opt.use_shared_T:
        netTP = netD
        netTQ = netD
    else:
        if opt.netT_model == 'proj64':
            netTP = SNResNetProjectionDiscriminator64(opt.ntf, opt.num_classes)
            netTQ = SNResNetProjectionDiscriminator64(opt.ntf, opt.num_classes)
        elif opt.netT_model == 'proj32':
            netTP = SNResNetProjectionDiscriminator32(opt.ntf, opt.num_classes)
            netTQ = SNResNetProjectionDiscriminator32(opt.ntf, opt.num_classes)
        else:
            raise NotImplementedError
if opt.netT != '':
    netTP.load_state_dict(torch.load(opt.netT))
print(netTP)

# loss functions
dis_criterion = nn.BCELoss()

# tensor placeholders
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(opt.batchSize)
real_label = torch.LongTensor(opt.batchSize)
fake_label = torch.LongTensor(opt.batchSize)
real_label_const = 1
fake_label_const = 0

# if using cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    netTP.cuda()
    netTQ.cuda()
    dis_criterion.cuda()
    input, dis_label, real_label, fake_label = input.cuda(), dis_label.cuda(), real_label.cuda(), fake_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
eval_noise = Variable(eval_noise)
dis_label = Variable(dis_label)

# noise for evaluation
eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
if opt.visualize_class_label >= 0:
    eval_label = np.ones(opt.batchSize, dtype=np.int) * opt.visualize_class_label
else:
    eval_label = np.random.randint(0, num_classes, opt.batchSize)
eval_onehot = np.zeros((opt.batchSize, num_classes))
eval_onehot[np.arange(opt.batchSize), eval_label] = 1
eval_noise_[np.arange(opt.batchSize), :num_classes] = eval_onehot[np.arange(opt.batchSize)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerTP = optim.Adam(netTP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerTQ = optim.Adam(netTQ.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

dset_name = os.path.split(opt.dataroot)[-1]
datafile = os.path.join(opt.dataroot, '..', f'{dset_name}_stats', dset_name)
sampler = ImageSampler(netG, opt)
get_metrics = prepare_inception_metrics(dataloader, datafile, False, opt.num_inception_images, no_is=False)

losses_D = []
losses_G = []
losses_IP = []
losses_IQ = []
losses_F = []
losses_IS_mean = []
losses_IS_std = []
for epoch in range(opt.niter):
    avg_loss_D = AverageMeter()
    avg_loss_G = AverageMeter()
    avg_loss_IP = AverageMeter()
    avg_loss_IQ = AverageMeter()
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        # train with real
        netD.zero_grad()
        real_cpu, label = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        with torch.no_grad():
            input.resize_as_(real_cpu).copy_(real_cpu)
            dis_label.resize_(batch_size).fill_(real_label_const)
            real_label.resize_(batch_size).copy_(label)
        dis_output, _ = netD(input)

        dis_errD_real = dis_criterion(dis_output, dis_label)
        errD_real = dis_errD_real
        errD_real.backward()
        D_x = dis_output.data.mean()

        # get fake
        if opt.shuffle_label == 'shuffle':
            label = label[torch.randperm(batch_size), ...].cpu().numpy()
        elif opt.shuffle_label == 'sample':
            label = np.random.randint(0, num_classes, batch_size)
        elif opt.shuffle_label == 'same':
            label = label.cpu().numpy()
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise.copy_(noise_.view(batch_size, nz, 1, 1))
        fake_label.resize_(batch_size).copy_(torch.from_numpy(label))
        fake = netG(noise)

        # train with fake
        dis_label.fill_(fake_label_const)
        dis_output, _ = netD(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        errD_fake = dis_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update T network
        ###########################
        # train with real
        y = real_label
        for _ in range(opt.n_update_mine):
            y_bar = y[torch.randperm(batch_size), ...]
            et = torch.mean(torch.exp(netTP(input, y_bar, 'P')))
            if netTP.ma_et_P is None:
                netTP.ma_et_P = et.detach().item()
            netTP.ma_et_P += opt.ma_rate * (et.detach().item() - netTP.ma_et_P)
            mi_P = torch.mean(netTP(input, y, 'P')) - torch.log(et) * et.detach() / netTP.ma_et_P
            loss_mine = -mi_P
            optimizerTP.zero_grad()
            loss_mine.backward()
            optimizerTP.step()

        # train with fake
        y = fake_label
        for _ in range(opt.n_update_mine):
            y_bar = y[torch.randperm(batch_size), ...]
            et = torch.mean(torch.exp(netTQ(fake.detach(), y_bar, 'Q')))
            if netTQ.ma_et_Q is None:
                netTQ.ma_et_Q = et.detach().item()
            netTQ.ma_et_Q += opt.ma_rate * (et.detach().item() - netTQ.ma_et_Q)
            mi_Q = torch.mean(netTQ(fake.detach(), y, 'Q')) - torch.log(et) * et.detach() / netTQ.ma_et_Q
            loss_mine = -mi_Q
            optimizerTQ.zero_grad()
            loss_mine.backward()
            optimizerTQ.step()

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label_const)  # fake labels are real for generator cost
        dis_output, _ = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)

        logP = netTP.log_prob(fake, fake_label, 'P')
        logQ = netTQ.log_prob(fake, fake_label, 'Q')
        logr = logP - logQ
        r = torch.exp(logr)

        if opt.f_div == 'kl':
            fr = r * logr
        elif opt.f_div == 'revkl':
            fr = -logr
        elif opt.f_div == 'pearson':
            fr = (r - 1).pow(2)
        elif opt.f_div == 'squared':
            fr = (r.sqrt() - 1).pow(2)
        elif opt.f_div == 'jsd':
            fr = -(r + 1.0) * torch.log(0.5 * r + 0.5) + r * logr
        elif opt.f_div == 'gan':
            fr = r * logr - (r + 1.0) * torch.log(r + 1.0)
        else:
            raise NotImplementedError
        fr = fr.mean()

        errG = dis_errG + opt.lambda_y * fr

        # adaptive
        if opt.adaptive:
            loss_G_u = dis_errG
            grad_u = autograd.grad(loss_G_u, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_y = autograd.grad(opt.lambda_y_grad * fr, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_u_flattened = torch.cat([g.view(-1) for g in grad_u])
            grad_y_flattened = torch.cat([g.view(-1) for g in grad_y])
            grad_u_norm = grad_u_flattened.pow(2).sum().sqrt()
            grad_y_norm = grad_y_flattened.pow(2).sum().sqrt()
            grad_a_ratio = torch.min(grad_u_norm, grad_y_norm) / grad_y_norm * opt.lambda_y
            for p, g_u, g_y in zip(netG.parameters(), grad_u, grad_y):
                p.grad = g_u + g_y * grad_a_ratio
        else:
            errG.backward()
        optimizerG.step()
        D_G_z2 = dis_output.data.mean()

        # compute the average loss
        avg_loss_G.update(errG.item(), batch_size)
        avg_loss_D.update(errD.item(), batch_size)
        avg_loss_IP.update(mi_P.item(), batch_size)
        avg_loss_IQ.update(mi_Q.item(), batch_size)

        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f IP: %.4f (%.4f) IQ: %.4f (%.4f)'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), avg_loss_D.avg, errG.item(), avg_loss_G.avg, D_x, D_G_z1, D_G_z2, mi_P.item(), avg_loss_IP.avg, mi_Q.item(), avg_loss_IQ.avg))
        if i % 100 == 0:
            vutils.save_image(
                real_cpu, '%s/real_samples.png' % opt.outf)
            # print('Label for eval = {}'.format(eval_label))
            fake = netG(eval_noise)
            vutils.save_image(
                fake.data,
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)
            )

    # compute metrics
    is_mean, is_std, fid = get_metrics(sampler, num_inception_images=opt.num_inception_images, num_splits=10,
                                       prints=True, use_torch=False)
    writer.add_scalar('Loss/G', avg_loss_G.avg, epoch)
    writer.add_scalar('Loss/D', avg_loss_D.avg, epoch)
    writer.add_scalar('Metric/IP', avg_loss_IP.avg, epoch)
    writer.add_scalar('Metric/IQ', avg_loss_IQ.avg, epoch)
    writer.add_scalar('Metric/FID', fid, epoch)
    writer.add_scalar('Metric/IS', is_mean, epoch)
    losses_G.append(avg_loss_G.avg)
    losses_D.append(avg_loss_D.avg)
    losses_IP.append(avg_loss_IP.avg)
    losses_IQ.append(avg_loss_IQ.avg)
    losses_F.append(fid)
    losses_IS_mean.append(is_mean)
    losses_IS_std.append(is_std)

    # do checkpointing
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    np.save(f'{opt.outf}/losses_G.npy', np.array(losses_G))
    np.save(f'{opt.outf}/losses_D.npy', np.array(losses_D))
    np.save(f'{opt.outf}/losses_IP.npy', np.array(losses_IP))
    np.save(f'{opt.outf}/losses_IQ.npy', np.array(losses_IQ))
    np.save(f'{opt.outf}/losses_F.npy', np.array(losses_F))
    np.save(f'{opt.outf}/losses_IS_mean.npy', np.array(losses_IS_mean))
    np.save(f'{opt.outf}/losses_IS_std.npy', np.array(losses_IS_std))
