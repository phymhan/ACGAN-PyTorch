## Unbiased Auxiliary Classifier GANs with MINE
This repo contains code for the [CVPR 2020 workshop](https://adv-workshop-2020.github.io) paper **Unbiased Auxiliary Classifier GANs with MINE**. [[pdf]](https://github.com/adv-workshop-2020/adv-workshop-2020.github.io/blob/master/short_papers/74.pdf) [[arXiv]](https://arxiv.org/pdf/2006.07567.pdf)

The code is largely based on the [PyTorch](https://pytorch.org/) implementation of [AC-GAN](https://github.com/clvrai/ACGAN-PyTorch)

### To reproduce results in the paper:
#### CIFAR10
To run AC-GAN:
```
python main.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --nz 100 --ny 10 --num_classes 10 --batchSize 256 --samplerBatchSize 256 --niter 200 --outf results/cifar10/ac --loss_type ac --label_rotation --visualize_class_label 0 --netD_model snres32 --manualSeed 0 --use_onehot_embed
```

To run TAC-GAN (reimplemented):
```
python main.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --nz 100 --ny 10 --num_classes 10 --batchSize 256 --samplerBatchSize 256 --niter 200 --outf results/cifar10/tac --loss_type tac --label_rotation --visualize_class_label 0 --netD_model snres32 --manualSeed 0 --use_onehot_embed
```

To run UAC-GAN (with MINE):
```
python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --nz 100 --ny 10 --num_classes 10 --batchSize 256 --samplerBatchSize 256 --niter 100 --no_ac_on_fake --outf results/cifar10/mine --use_shared_T --netD_model proj32 --loss_type mine --label_rotation --visualize_class_label 0 --netT_model proj32 --use_cy --lambda_mi 0.1 --manualSeed 0 --use_onehot_embed
```
