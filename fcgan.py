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
import utils
from network2 import _netG_Res32, _netD_Res32, loss_hinge_gen, loss_idt_gen
from folder import ImageFolder
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from inception import prepare_inception_metrics, prepare_data_statistics
import torch.nn.functional as F
import functools
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--samplerBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ny', type=int, default=0, help='size of the latent embedding vector for y')
parser.add_argument('--use_onehot_embed', action='store_true', help='use onehot embedding in G?')
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
parser.add_argument('--visualize_class_label', type=int, default=-1, help='if < 0, random int')
parser.add_argument('--lambda_mi', type=float, default=1.)
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--n_update_mine', type=int, default=1, help='how many updates on mine in each iteration')
parser.add_argument('--download_dset', action='store_true')
parser.add_argument('--num_inception_images', type=int, default=10000)
parser.add_argument('--add_eta', action='store_true')
parser.add_argument('--no_sn_dis', action='store_true')
parser.add_argument('--emb_init_zero', action='store_true')
parser.add_argument('--use_softmax', action='store_true')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--lambda_T', type=float, default=0.01)
parser.add_argument('--lambda_T_decay', type=float, default=0)
parser.add_argument('--label_rotation', action='store_true')
parser.add_argument('--disable_cudnn_benchmark', action='store_true')
parser.add_argument('--feature_save', action='store_true')
parser.add_argument('--feature_save_every', type=int, default=1)
parser.add_argument('--feature_num_batches', type=int, default=1)
parser.add_argument('--store_linear', action='store_true')
parser.add_argument('--sample_trunc_normal', action='store_true')
parser.add_argument('--separate', action='store_true')
parser.add_argument('--mi_type_p', type=str, default='ce')
parser.add_argument('--mi_type_q', type=str, default='ce')
parser.add_argument('--f_loss', type=str, default='identity')

opt = parser.parse_args()
print_options(parser, opt)

# specify the gpu id if using only 1 gpu
# if opt.ngpu == 1:
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

outff = os.path.join(opt.outf, 'features')
if opt.feature_save or opt.store_linear:
    utils.mkdirs(outff)

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

# dataset
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
nz = int(opt.nz)
ny = int(opt.num_classes) if opt.ny == 0 else int(opt.ny)  # embedding dim same as onehot embedding by default
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3

# Define the generator and initialize the weights
netG = _netG_Res32(ngpu, nz, ny, num_classes, one_hot=opt.use_onehot_embed)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Define the discriminator and initialize the weights
netD = _netD_Res32(ngpu, num_classes, mi_type_p=opt.mi_type_p, mi_type_q=opt.mi_type_q,
                   add_eta=opt.add_eta, no_sn_dis=opt.no_sn_dis, use_softmax=opt.use_softmax)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# loss functions
dis_criterion = nn.BCEWithLogitsLoss()
aux_criterion = nn.CrossEntropyLoss()
if opt.f_loss == 'hinge':
    f_criterion = loss_hinge_gen
elif opt.f_loss == 'identity':
    f_criterion = loss_idt_gen
else:
    raise NotImplementedError

# tensor placeholders
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.randn(opt.batchSize, nz, requires_grad=False)
eval_noise = torch.randn(opt.batchSize, nz, requires_grad=False)
fake_label = torch.LongTensor(opt.batchSize)
dis_label = torch.FloatTensor(opt.batchSize)
real_label_const = 1
fake_label_const = 0

feature_eval_noises = []
feature_eval_labels = []
if opt.feature_save:
    for i in range(opt.feature_num_batches):
        z = torch.randn(opt.batchSize, nz, requires_grad=False).normal_(0, 1)
        y = torch.LongTensor(opt.batchSize).random_(0, num_classes)
        if opt.sample_trunc_normal:
            utils.truncated_normal_(z, 0, 1)
        feature_eval_noises.append(z.cuda())
        feature_eval_labels.append(y.cuda())

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
losses_IP = []
losses_IQ = []
losses_A = []
losses_F = []
losses_IS_mean = []
losses_IS_std = []
feature_batches = []
for epoch in range(opt.niter):
    avg_loss_D = AverageMeter()
    avg_loss_G = AverageMeter()
    avg_loss_A = AverageMeter()
    avg_loss_IP = AverageMeter()
    avg_loss_IQ = AverageMeter()
    avg_loss_f = AverageMeter()
    feature_batch_counter = 0
    for i, data in enumerate(dataloader, 0):
        # if save_features, save at the beginning of an epoch
        if opt.feature_save and epoch % opt.feature_save_every == 0 and feature_batch_counter < opt.feature_num_batches:
            if len(feature_batches) < opt.feature_num_batches:
                eval_x, eval_y = data
                eval_x = eval_x.cuda()
                feature_batches.append((eval_x, eval_y))
            # feature for real
            eval_x, eval_y = feature_batches[feature_batch_counter]
            with torch.no_grad():
                eval_f = netD.get_feature(eval_x)
            utils.save_features(eval_f.cpu().numpy(),
                                os.path.join(outff, f'real_epoch_{epoch}_batch_{feature_batch_counter}_f.npy'))
            utils.save_features(eval_y.cpu().numpy(),
                                os.path.join(outff, f'real_epoch_{epoch}_batch_{feature_batch_counter}_y.npy'))
            # feature for fake
            with torch.no_grad():
                eval_x = netG(feature_eval_noises[feature_batch_counter], feature_eval_labels[feature_batch_counter])
                eval_y = feature_eval_labels[feature_batch_counter]
                eval_f = netD.get_feature(eval_x)
            utils.save_features(eval_f.cpu().numpy(),
                                os.path.join(outff, f'fake_epoch_{epoch}_batch_{feature_batch_counter}_f.npy'))
            utils.save_features(eval_y.cpu().numpy(),
                                os.path.join(outff, f'fake_epoch_{epoch}_batch_{feature_batch_counter}_y.npy'))
            feature_batch_counter += 1
            continue

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        # train with real
        netD.zero_grad()
        real_cpu, label = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu, label = real_cpu.cuda(), label.cuda()
        with torch.no_grad():
            input.resize_as_(real_cpu).copy_(real_cpu)
            dis_label.resize_(batch_size).fill_(real_label_const)
        dis_output, aux_output = netD(input)
        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, label)
        (dis_errD_real + aux_errD_real).backward()
        errD_real = dis_errD_real
        D_x = torch.sigmoid(dis_output).data.mean()

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, label)

        # get fake
        fake_label.resize_(batch_size).random_(0, num_classes)
        noise.resize_(batch_size, nz).normal_(0, 1)
        fake = netG(noise, fake_label)

        # train with fake
        dis_label.resize_(batch_size).fill_(fake_label_const)
        dis_output, _ = netD(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        errD_fake = dis_errD_fake
        errD_fake.backward()
        D_G_z1 = torch.sigmoid(dis_output).data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Mutual Information
        ###########################
        real_label = label
        # train with real
        y = real_label
        for _ in range(opt.n_update_mine):
            optimizerD.zero_grad()
            if opt.mi_type_p == 'ce':
                errD_mi_P = -netD(input, y, 'P').mean()
                mi_P = np.log(num_classes) - errD_mi_P
            elif opt.mi_type_p == 'mine':
                y_bar = y[torch.randperm(batch_size), ...]
                tp = netD(input, y, 'P')
                tp_bar = netD(input, y_bar, 'P')
                tp_bar_max = tp_bar.max().detach()
                mi_P = torch.mean(tp) - torch.log(torch.mean(torch.exp(tp_bar - tp_bar_max))) - tp_bar_max
                errD_mi_P = -mi_P
            elif opt.mi_type_p == 'eta':
                y_bar = y[torch.randperm(batch_size), ...]
                tp = netD(input, y, 'P')
                tp_bar = netD(input, y_bar, 'P')
                mi_P = torch.mean(tp) - torch.mean(torch.exp(tp_bar))
                errD_mi_P = -mi_P
            errD_mi_P.backward()
            optimizerD.step()

        # train with fake
        y = fake_label
        for _ in range(opt.n_update_mine):
            optimizerD.zero_grad()
            if opt.mi_type_q == 'ce':
                errD_mi_q = -netD(fake.detach(), y, 'Q').mean()
                mi_Q = np.log(num_classes) - errD_mi_Q
            elif opt.mi_type_q == 'mine':
                y_bar = y[torch.randperm(batch_size), ...]
                tq = netD(fake.detach(), y, 'Q')
                tq_bar = netD(fake.detach(), y_bar, 'Q')
                tq_bar_max = tq_bar.max().detach()
                mi_Q = torch.mean(tq) - torch.log(torch.mean(torch.exp(tq_bar - tq_bar_max))) - tq_bar_max
                errD_mi_Q = -mi_Q
            elif opt.mi_type_q == 'eta':
                y_bar = y[torch.randperm(batch_size), ...]
                tq = netD(fake.detach(), y, 'Q')
                tq_bar = netD(fake.detach(), y_bar, 'Q')
                mi_Q = torch.mean(tq) - torch.mean(torch.exp(tq_bar))
                errD_mi_Q = -mi_Q
            errD_mi_Q.backward()
            optimizerD.step()

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label_const)  # fake labels are real for generator cost
        dis_output, _ = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)
        logP = netD.log_prob(fake, fake_label, 'P')
        logQ = netD.log_prob(fake, fake_label, 'Q')
        f_div = f_criterion(logQ - logP)

        errG = dis_errG + opt.lambda_mi * f_div
        errG.backward()
        optimizerG.step()
        D_G_z2 = torch.sigmoid(dis_output).data.mean()

        # compute the average loss
        avg_loss_G.update(errG.item(), batch_size)
        avg_loss_D.update(errD.item(), batch_size)
        avg_loss_A.update(accuracy, batch_size)
        avg_loss_IP.update(mi_P.item(), batch_size)
        avg_loss_IQ.update(mi_Q.item(), batch_size)
        avg_loss_f.update(f_div.item(), batch_size)

        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f IP: %.4f (%.4f) IQ: %.4f (%.4f) f-div: %.4f (%.4f) Acc: %.4f (%.4f)'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), avg_loss_D.avg, errG.item(), avg_loss_G.avg, D_x, D_G_z1, D_G_z2,
                 mi_P.item(), avg_loss_IP.avg, mi_Q.item(), avg_loss_IQ.avg, f_div.item(), avg_loss_f.avg,
                 accuracy, avg_loss_A.avg))
        if i % 100 == 0:
            vutils.save_image(
                utils.normalize(real_cpu), '%s/real_samples.png' % opt.outf)
            # print('Label for eval = {}'.format(eval_label))
            fake = netG(eval_noise, eval_label)
            vutils.save_image(
                utils.normalize(fake.data),
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)
            )

    # update eval_label
    if opt.visualize_class_label >= 0 and opt.label_rotation:
        eval_label_const = (eval_label_const + 1) % num_classes
        eval_label.data.fill_(eval_label_const)

    # compute metrics
    is_mean, is_std, fid = get_metrics(sampler, num_inception_images=opt.num_inception_images, num_splits=10,
                                       prints=True, use_torch=False)
    if opt.store_linear:
        names = netD.get_linear_name()
        params = netD.get_linear()
        for (name, param) in zip(netD.get_linear_name(), netD.get_linear()):
            if param is not None:
                np.save(os.path.join(outff, f'{name}_epoch_{epoch}.npy'), param)
    writer.add_scalar('Loss/G', avg_loss_G.avg, epoch)
    writer.add_scalar('Loss/D', avg_loss_D.avg, epoch)
    writer.add_scalar('Loss/f_div', avg_loss_f.avg, epoch)
    writer.add_scalar('Metric/Aux', avg_loss_A.avg, epoch)
    writer.add_scalar('Metric/IP', avg_loss_IP.avg, epoch)
    writer.add_scalar('Metric/IQ', avg_loss_IQ.avg, epoch)
    writer.add_scalar('Metric/FID', fid, epoch)
    writer.add_scalar('Metric/IS', is_mean, epoch)
    losses_G.append(avg_loss_G.avg)
    losses_D.append(avg_loss_D.avg)
    losses_A.append(avg_loss_A.avg)
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
    np.save(f'{opt.outf}/losses_A.npy', np.array(losses_A))
    np.save(f'{opt.outf}/losses_IP.npy', np.array(losses_IP))
    np.save(f'{opt.outf}/losses_IQ.npy', np.array(losses_IQ))
    np.save(f'{opt.outf}/losses_F.npy', np.array(losses_F))
    np.save(f'{opt.outf}/losses_IS_mean.npy', np.array(losses_IS_mean))
    np.save(f'{opt.outf}/losses_IS_std.npy', np.array(losses_IS_std))
