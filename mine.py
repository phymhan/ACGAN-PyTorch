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
from utils import weights_init, compute_acc, AverageMeter, ImageSampler
from network import _netG, _netD, _netT, _netD_CIFAR10, _netG_CIFAR10, _netT_concat_CIFAR10, _netDT_CIFAR10
from network import SNResNetProjectionDiscriminator64, SNResNetProjectionDiscriminator32
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
parser.add_argument('--lambda_mi_grad', type=float, default=1.)
parser.add_argument('--lambda_mi', type=float, default=1.)
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--adaptive_grad', type=str, default='dc', help='[d | c | dc]')
parser.add_argument('--n_update_mine', type=int, default=1, help='how many updates on mine in each iteration')
parser.add_argument('--download_dset', action='store_true')
parser.add_argument('--num_inception_images', type=int, default=10000)
parser.add_argument('--use_shared_T', action='store_true')
parser.add_argument('--netT_model', type=str, default='concat')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

opt = parser.parse_args()
print(opt)

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
elif opt.dataset == 'cifar10':
    if opt.use_shared_T:
        netD = _netDT_CIFAR10(ngpu, num_classes)
    else:
        netD = _netD_CIFAR10(ngpu, num_classes, tac=False)
elif opt.dataset == 'mnist':
    if opt.use_shared_T:
        netD = _netDT_CIFAR10(ngpu, num_classes)
    else:
        netD = _netD_CIFAR10(ngpu, num_classes, tac=False)
else:
    raise NotImplementedError
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Define the statistics network and initialize the weights
if opt.dataset == 'imagenet':
    netT = _netT(ngpu)
else:
    if opt.use_shared_T:
        netT = netD
    else:
        if opt.netT_model == 'concat':
            netT = _netT_concat_CIFAR10(ngpu)
            netT.apply(weights_init)
        elif opt.netT_model == 'proj64':
            netT = SNResNetProjectionDiscriminator64(opt.ntf, opt.num_classes)
            # netT._initialize()
        elif opt.netT_model == 'proj32':
            netT = SNResNetProjectionDiscriminator32(opt.ntf, opt.num_classes)
            # netT._initialize()
if opt.netT != '':
    netT.load_state_dict(torch.load(opt.netT))
print(netT)

# loss functions
dis_criterion = nn.BCELoss()
aux_criterion = nn.CrossEntropyLoss()

# tensor placeholders
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(opt.batchSize)
aux_label = torch.LongTensor(opt.batchSize)
aux_label_bar = torch.LongTensor(opt.batchSize)
real_label = 1
fake_label = 0

# if using cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    netT.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    input, dis_label, aux_label, aux_label_bar = input.cuda(), dis_label.cuda(), aux_label.cuda(), aux_label_bar.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
eval_noise = Variable(eval_noise)
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)
aux_label_bar = Variable(aux_label_bar)
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
optimizerT = optim.Adam(netT.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

dset_name = os.path.split(opt.dataroot)[-1]
datafile = os.path.join(opt.dataroot, '..', f'{dset_name}_stats', dset_name)
sampler = ImageSampler(netG, opt)
get_metrics = prepare_inception_metrics(dataloader, datafile, False, opt.num_inception_images, no_is=False)

losses_D = []
losses_G = []
losses_A = []
losses_M = []
losses_F = []
losses_I_mean = []
losses_I_std = []
for epoch in range(opt.niter):
    avg_loss_D = AverageMeter()
    avg_loss_G = AverageMeter()
    avg_loss_A = AverageMeter()
    avg_loss_M = AverageMeter()
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, label = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        with torch.no_grad():
            input.resize_as_(real_cpu).copy_(real_cpu)
            dis_label.resize_(batch_size).fill_(real_label)
            aux_label.resize_(batch_size).copy_(label)
        dis_output, aux_output = netD(input)

        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.data.mean()

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label)

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        label = np.random.randint(0, num_classes, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise.copy_(noise_.view(batch_size, nz, 1, 1))
        aux_label.resize_(batch_size).copy_(torch.from_numpy(label))
        fake = netG(noise)

        # train with fake
        dis_label.fill_(fake_label)
        dis_output, aux_output = netD(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update T network
        ###########################
        # label_bar = np.random.randint(0, num_classes, batch_size)
        # aux_label_bar.resize_(batch_size).copy_(torch.from_numpy(label_bar))
        # y, y_bar = aux_label, aux_label_bar
        y = aux_label
        for _ in range(opt.n_update_mine):
            y_bar = y[torch.randperm(batch_size), ...]
            et = torch.mean(torch.exp(netT(fake.detach(), y_bar)))
            if netT.ma_et is None:
                netT.ma_et = et.detach().item()
            netT.ma_et += opt.ma_rate * (et.detach().item() - netT.ma_et)
            mi = torch.mean(netT(fake.detach(), y)) - torch.log(et) * et.detach() / netT.ma_et
            loss_mine = -mi
            optimizerT.zero_grad()
            loss_mine.backward()
            optimizerT.step()

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label)  # fake labels are real for generator cost
        dis_output, aux_output = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        mi = torch.mean(netT(fake, y)) - torch.log(torch.mean(torch.exp(netT(fake, y_bar))))
        errG = dis_errG + aux_errG + opt.lambda_mi * mi

        # adaptive
        if opt.adaptive:
            loss_G_u = 0.
            if 'd' in opt.adaptive_grad:
                loss_G_u += dis_errG
            if 'c' in opt.adaptive_grad:
                loss_G_u += aux_errG
            grad_u = autograd.grad(loss_G_u, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_m = autograd.grad(opt.lambda_mi_grad * mi, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_d = autograd.grad(dis_errG, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_c = autograd.grad(aux_errG, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_u_flattened = torch.cat([g.view(-1) for g in grad_u])
            grad_m_flattened = torch.cat([g.view(-1) for g in grad_m])
            grad_u_norm = grad_u_flattened.pow(2).sum().sqrt()
            grad_m_norm = grad_m_flattened.pow(2).sum().sqrt()
            grad_a_ratio = torch.min(grad_u_norm, grad_m_norm) / grad_m_norm * opt.lambda_mi
            for p, g_d, g_c, g_m in zip(netG.parameters(), grad_d, grad_c, grad_m):
                p.grad = g_d + g_c + g_m * grad_a_ratio
        else:
            errG.backward()
        optimizerG.step()
        D_G_z2 = dis_output.data.mean()

        # compute the average loss
        avg_loss_G.update(errG.item(), batch_size)
        avg_loss_D.update(errD.item(), batch_size)
        avg_loss_A.update(accuracy, batch_size)
        avg_loss_M.update(mi.item(), batch_size)

        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f) MI: %.4f (%.4f)'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), avg_loss_D.avg, errG.item(), avg_loss_G.avg, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A.avg, mi.item(), avg_loss_M.avg))
        if i % 100 == 0:
            vutils.save_image(
                real_cpu, '%s/real_samples.png' % opt.outf)
            print('Label for eval = {}'.format(eval_label))
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
    writer.add_scalar('Metric/Aux', avg_loss_A.avg, epoch)
    writer.add_scalar('Metric/MI', avg_loss_M.avg, epoch)
    writer.add_scalar('Metric/FID', fid, epoch)
    writer.add_scalar('Metric/IS', is_mean, epoch)
    losses_G.append(avg_loss_G.avg)
    losses_D.append(avg_loss_D.avg)
    losses_A.append(avg_loss_A.avg)
    losses_M.append(avg_loss_M.avg)
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
    np.save(f'{opt.outf}/losses_M.npy', np.array(losses_M))
    np.save(f'{opt.outf}/losses_F.npy', np.array(losses_F))
    np.save(f'{opt.outf}/losses_I_mean.npy', np.array(losses_I_mean))
    np.save(f'{opt.outf}/losses_I_std.npy', np.array(losses_I_std))
