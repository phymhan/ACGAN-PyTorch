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
from network import _netG, _netG_CIFAR10, EmbeddingNet, ReconstructorConcat, ReconstructorSiamese
from folder import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from inception import prepare_inception_metrics
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--lambda_shift', type=float, default=1.0)
parser.add_argument('--download_dset', action='store_true')
parser.add_argument('--num_inception_images', type=int, default=10000)
parser.add_argument('--netR_model', type=str, default='concat', help='[concat | siamese]')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--epsilon', type=float, default=1.0)

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

# Define matrix A
netA = EmbeddingNet(nz, num_classes)

# Define Reconstructor
if opt.netR_model == 'concat':
    netR = ReconstructorConcat(opt.ndf, num_classes)
elif opt.netR_model == 'siamese':
    netR = ReconstructorSiamese(opt.ndf, num_classes)
else:
    raise NotImplementedError
print(netR)

# loss functions
class_criterion = nn.CrossEntropyLoss()
shift_criterion = nn.L1Loss()

# tensor placeholders
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)

# if using cuda
if opt.cuda:
    netG.cuda()
    netA.cuda()
    netR.cuda()
    class_criterion.cuda()
    shift_criterion.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# define variables
noise = Variable(noise)
# noise for evaluation
eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))

# setup optimizer
optimizerA = optim.Adam(netA.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

dset_name = os.path.split(opt.dataroot)[-1]
datafile = os.path.join(opt.dataroot, '..', f'{dset_name}_stats', dset_name)
sampler = ImageSampler(netG, opt)
get_metrics = prepare_inception_metrics(dataloader, datafile, False, opt.num_inception_images, no_is=False)

losses_class = []
losses_shift = []
batch_size = opt.batchSize
avg_loss_class = AverageMeter()
avg_loss_shift = AverageMeter()
avg_loss_acc = AverageMeter()
for i in range(opt.niter):
    # sample z, k, eps
    noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
    noise_ = np.random.normal(0, 1, (batch_size, nz))
    noise_ = (torch.from_numpy(noise_))
    noise.copy_(noise_.view(batch_size, nz, 1, 1))
    label = torch.LongTensor(batch_size).random_(0, opt.num_classes).cuda()
    shift = torch.FloatTensor(batch_size).uniform_(-opt.epsilon, opt.epsilon).cuda()

    noise_shifted = noise + shift.view(batch_size, 1, 1, 1) * netA(label).view(batch_size, nz, 1, 1)

    netA.zero_grad()
    netR.zero_grad()
    x1 = netG(noise)
    x2 = netG(noise_shifted)

    pred_class, pred_shift = netR(x1, x2)
    loss_class = class_criterion(pred_class, label)
    loss_shift = shift_criterion(pred_shift, shift)
    loss = loss_class + loss_shift
    loss.backward()
    optimizerA.step()
    optimizerR.step()

    accuracy = compute_acc(pred_class, label)

    avg_loss_class.update(loss_class.item(), batch_size)
    avg_loss_shift.update(loss_shift.item(), batch_size)
    avg_loss_acc.update(accuracy, batch_size)

    if i % 10 == 0:
        print('[%d/%d] Loss_Class: %.4f (%.4f) Loss_Shift: %.4f (%.4f) Acc_Class: %.4f (%.4f)'
              % (i, opt.niter, loss_class.item(), avg_loss_class.avg, loss_shift.item(), avg_loss_shift.avg, accuracy, avg_loss_acc.avg))
        writer.add_scalar('Loss/Class', avg_loss_class.avg, i)
        writer.add_scalar('Loss/Shift', avg_loss_shift.avg, i)
        writer.add_scalar('Acc/Class', avg_loss_acc.avg, i)

    # do checkpointing
    if i % 10 == 0:
        torch.save(netA.state_dict(), '%s/netA_iter_%d.pth' % (opt.outf, i))
        torch.save(netR.state_dict(), '%s/netR_iter_%d.pth' % (opt.outf, i))
