import torch
import numpy as np
from torch.nn import init
import os
import sys
import pdb

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Embedding') != -1:
        # m.weight.data.normal_(1.0, 0.02)
        init.xavier_uniform_(m.weight.data)


# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vec2sca_avg = 0
        self.vec2sca_val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if torch.is_tensor(self.val) and torch.numel(self.val) != 1:
            self.avg[self.count == 0] = 0
            self.vec2sca_avg = self.avg.sum() / len(self.avg)
            self.vec2sca_val = self.val.sum() / len(self.val)


class ImageSampler:
    def __init__(self, G, opt):
        self.G = G
        self.noise = torch.FloatTensor(opt.samplerBatchSize, opt.nz)
        self.label = torch.LongTensor(opt.samplerBatchSize)
        self.batchSize = opt.batchSize
        self.opt = opt

    def __iter__(self):
        return self

    def __next__(self):
        # self.noise.normal_(0, 1)
        # label = np.random.randint(0, self.opt.num_classes, self.batchSize)
        # noise_ = np.random.normal(0, 1, (self.batchSize, self.opt.nz))
        # class_onehot = np.zeros((self.batchSize, self.opt.num_classes))
        # class_onehot[np.arange(self.batchSize), label] = 1
        # noise_[np.arange(self.batchSize), :self.opt.num_classes] = class_onehot[np.arange(self.batchSize)]
        # noise_ = (torch.from_numpy(noise_))
        # self.noise.copy_(noise_.view(self.batchSize, self.opt.nz, 1, 1))
        # self.label.resize_(self.batchSize).copy_(torch.from_numpy(label))
        # fake = self.G(self.noise.cuda())
        self.noise.normal_(0, 1)
        self.label.random_(0, self.opt.num_classes)
        return self.G(self.noise.cuda(), self.label.cuda()), self.label


def print_options(parser, opt):
    message = ''
    message += '--------------- Options -----------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = opt.outf
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

    # save command to disk
    file_name = os.path.join(expr_dir, 'cmd.sh')
    with open(file_name, 'wt') as cmd_file:
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            cmd_file.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        cmd_file.write(' '.join(sys.argv))
        cmd_file.write('\n')


def set_onehot(noise, label, nclass):
    bs = noise.size(0)
    nz = noise.size(1)
    label = np.ones(bs, dtype=np.int) * label
    noise_numpy = noise.view(bs, nz).cpu().numpy()
    onehot = np.zeros((bs, nclass))
    onehot[np.arange(bs), label] = 1
    noise_numpy[np.arange(bs), :nclass] = onehot[np.arange(bs)]
    noise.data.copy_(torch.from_numpy(noise_numpy).view(bs, nz, 1, 1))
    return noise


# borrowed from BigGAN
class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# save features
def save_features(feat, filename):
    np.save(filename, feat)


def normalize(tensor):
    # normalize [-1, 1] to [0, 1]
    return (tensor + 1.) / 2.
