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
from network import _netG, _netD, _netT, _netD_CIFAR10, _netD_SNRes32, _netG_CIFAR10, _netT_concat_CIFAR10, _netDT_CIFAR10
from network import SNResNetProjectionDiscriminator32, _netDT_SNResProj32
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
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--samplerBatchSize', type=int, default=256, help='input batch size')
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
parser.add_argument('--loss_type', type=str, default='mine', help='[mine | eta]')
parser.add_argument('--visualize_class_label', type=int, default=-1, help='if < 0, random int')
parser.add_argument('--ma_rate', type=float, default=0.001)
parser.add_argument('--lambda_mi', type=float, default=1.)
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--adaptive_grad', type=str, default='dc', help='[d | c | dc]')
parser.add_argument('--n_update_mine', type=int, default=1, help='how many updates on mine in each iteration')
parser.add_argument('--download_dset', action='store_true')
parser.add_argument('--num_inception_images', type=int, default=10000)
parser.add_argument('--use_shared_T', action='store_true')
parser.add_argument('--use_cy', action='store_true')
parser.add_argument('--no_sn_emb_l', action='store_true')
parser.add_argument('--no_sn_emb_c', action='store_true')
parser.add_argument('--emb_init_zero', action='store_true')
parser.add_argument('--softmax_T', action='store_true')
parser.add_argument('--netD_model', type=str, default='basic', help='[basic | proj32]')
parser.add_argument('--netT_model', type=str, default='concat', help='[concat | proj32 | proj64]')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--bnn_dropout', type=float, default=0.)
# parser.add_argument('--shuffle_label', type=str, default='uniform', help='[uniform | shuffle | same]')
parser.add_argument('--weighted_mine_loss', action='store_true', default=False)
parser.add_argument('--label_rotation', action='store_true')
parser.add_argument('--eps', type=float, default=0., help='eps added in log')
parser.add_argument('--disable_cudnn_benchmark', action='store_true')
parser.add_argument('--no_ac_on_fake', action='store_true')
parser.add_argument('--feature_save', action='store_true')
parser.add_argument('--feature_save_every', type=int, default=1)
parser.add_argument('--feature_num_batches', type=int, default=1)
parser.add_argument('--no_ma_trick', action='store_true')
parser.add_argument('--tbar_save', action='store_true')
parser.add_argument('--tbar_save_every', type=int, default=1)
parser.add_argument('--tbar_num_batches', type=int, default=1)
parser.add_argument('--no_neg_log_py', action='store_true')
parser.add_argument('--weighted_D_loss', action='store_true', help='If True, lambda_tac is also applied to D')
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
if opt.feature_save:
    utils.mkdirs(outff)

outft = os.path.join(opt.outf, 'statistics')
if opt.tbar_save:
    utils.mkdirs(outft)

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

# dataset
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

neg_log_py = np.log(num_classes)

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
    netD = _netD(ngpu, num_classes)
elif opt.dataset == 'mnist' or opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
    netD = _netDT_SNResProj32(opt.ndf, opt.num_classes, use_cy=opt.use_cy, dropout=opt.bnn_dropout,
                              sn_emb_l=not opt.no_sn_emb_l, sn_emb_c=not opt.no_sn_emb_c,
                              init_zero=opt.emb_init_zero, softmax=opt.softmax_T, eta=opt.loss_type == 'eta',
                              no_neg_log_py=opt.no_neg_log_py)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Define the statistics network and initialize the weights
netT = netD

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

feature_eval_noises = []
feature_eval_labels = []
if opt.feature_save:
    for i in range(opt.feature_num_batches):
        z = torch.randn(opt.batchSize, nz, requires_grad=False).cuda()
        y = torch.LongTensor(opt.batchSize).cuda()
        feature_eval_noises.append(z.normal_(0, 1))
        feature_eval_labels.append(y.random_(0, num_classes))

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
feature_batches = []
for epoch in range(opt.niter):
    avg_loss_D = AverageMeter()
    avg_loss_G = AverageMeter()
    avg_loss_A = AverageMeter()
    avg_loss_M = AverageMeter()
    feature_batch_counter = 0
    tbar_batch_counter = 0
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
        dis_output, aux_output = netD(input)

        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, label)
        if opt.weighted_D_loss:
            errD_real = dis_errD_real + aux_errD_real * opt.lambda_mi
        else:
            errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = torch.sigmoid(dis_output).data.mean()

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, label)

        # get fake
        fake_label.resize_(batch_size).random_(0, num_classes)
        noise.resize_(batch_size, nz).normal_(0, 1)
        fake = netG(noise, fake_label)

        # train with fake
        dis_label.resize_(batch_size).fill_(fake_label_const)
        dis_output, aux_output = netD(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = 0. if opt.no_ac_on_fake else aux_criterion(aux_output, fake_label)
        if opt.weighted_D_loss:
            errD_fake = dis_errD_fake + aux_errD_fake * opt.lambda_mi
        else:
            errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = torch.sigmoid(dis_output).data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update T network
        ###########################
        # label_bar = np.random.randint(0, num_classes, batch_size)
        # aux_label_bar.resize_(batch_size).copy_(torch.from_numpy(label_bar))
        # y, y_bar = aux_label, aux_label_bar
        y = fake_label
        for _ in range(opt.n_update_mine):
            y_bar = y[torch.randperm(batch_size), ...]
            if opt.loss_type == 'eta':
                mi = torch.mean(netT(fake.detach(), y)) - torch.mean(torch.exp(netT(fake.detach(), y_bar))) + 1
            else:
                if opt.no_ma_trick:
                    tbar = netT(fake.detach(), y_bar)
                    tbar_max = tbar.max().detach()
                    log_mean_exp_tbar = tbar_max + torch.log(torch.mean(torch.exp(tbar - tbar_max)))
                    mi = torch.mean(netT(fake.detach(), y)) - log_mean_exp_tbar
                else:
                    et = torch.mean(torch.exp(netT(fake.detach(), y_bar)))
                    if netT.ma_et is None:
                        netT.ma_et = et.detach().item()
                    netT.ma_et += opt.ma_rate * (et.detach().item() - netT.ma_et)
                    mi = torch.mean(netT(fake.detach(), y)) - torch.log(et + opt.eps) * et.detach() / netT.ma_et
            loss_mine = -mi * opt.lambda_mi if opt.weighted_mine_loss else -mi
            optimizerT.zero_grad()
            loss_mine.backward()
            optimizerT.step()

        if opt.tbar_save and epoch % opt.tbar_save_every == 0 and tbar_batch_counter < opt.tbar_num_batches:
            # save T, T_bar, mean(T), log_sum_exp(T_bar)
            with torch.no_grad():
                t = netT.log_prob(fake.detach(), y)
                tbar = netT.log_prob(fake.detach(), y_bar)
                mean_t = torch.mean(t)
                tbar_max = tbar.max().detach()
                lse_t = tbar_max + torch.log(torch.sum(torch.exp(tbar - tbar_max)))
                utils.save_features(t.cpu().numpy(),
                                    os.path.join(outft, f'fake_epoch_{epoch}_batch_{tbar_batch_counter}_t.npy'))
                utils.save_features(mean_t.cpu().numpy(),
                                    os.path.join(outft, f'fake_epoch_{epoch}_batch_{tbar_batch_counter}_meant.npy'))
                utils.save_features(tbar.cpu().numpy(),
                                    os.path.join(outft, f'fake_epoch_{epoch}_batch_{tbar_batch_counter}_tbar.npy'))
                utils.save_features(lse_t.cpu().numpy(),
                                    os.path.join(outft, f'fake_epoch_{epoch}_batch_{tbar_batch_counter}_lset.npy'))
            tbar_batch_counter += 1

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label_const)  # fake labels are real for generator cost
        dis_output, aux_output = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, fake_label)
        y_bar = y[torch.randperm(batch_size), ...]
        mi_errG = torch.mean(netT(fake, y)) - torch.log(torch.mean(torch.exp(netT(fake, y_bar)))+opt.eps)
        if opt.weighted_D_loss:
            errG = dis_errG + (aux_errG + mi_errG) * opt.lambda_mi
        else:
            errG = dis_errG + aux_errG + mi_errG * opt.lambda_mi

        # adaptive
        if opt.adaptive:
            loss_G_u = 0.
            if 'd' in opt.adaptive_grad:
                loss_G_u += dis_errG
            if 'c' in opt.adaptive_grad:
                loss_G_u += aux_errG
            grad_u = autograd.grad(loss_G_u, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_m = autograd.grad(opt.lambda_mi * mi_errG, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_d = autograd.grad(dis_errG, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_c = autograd.grad(aux_errG, netG.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
            grad_u_flattened = torch.cat([g.view(-1) for g in grad_u])
            grad_m_flattened = torch.cat([g.view(-1) for g in grad_m])
            grad_u_norm = grad_u_flattened.pow(2).sum().sqrt()
            grad_m_norm = grad_m_flattened.pow(2).sum().sqrt()
            grad_m_ratio = torch.min(grad_u_norm, grad_m_norm) / grad_m_norm
            for p, g_d, g_c, g_m in zip(netG.parameters(), grad_d, grad_c, grad_m):
                p.grad = g_d + g_c + g_m * grad_m_ratio
        else:
            errG.backward()
        optimizerG.step()
        D_G_z2 = torch.sigmoid(dis_output).data.mean()

        # compute the average loss
        avg_loss_G.update(errG.item(), batch_size)
        avg_loss_D.update(errD.item(), batch_size)
        avg_loss_A.update(accuracy, batch_size)
        avg_loss_M.update(mi.item(), batch_size)

        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f) MI: %.4f (%.4f)'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), avg_loss_D.avg,
                 errG.item(), avg_loss_G.avg,
                 D_x, D_G_z1, D_G_z2,
                 accuracy, avg_loss_A.avg,
                 mi.item(), avg_loss_M.avg))
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
