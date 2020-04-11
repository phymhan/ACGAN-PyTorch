#CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ls_csz --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy
#
CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ls_cnz --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy --no_sn_emb_c
#
CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ln_cnz --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l
#
#CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ln_csz --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy --no_sn_emb_l
#

CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ls_cs_ --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy

CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ls_cn_ --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy --no_sn_emb_c

CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ln_cn_ --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l

CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ln_cs_ --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy --no_sn_emb_l

CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ls_ --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32

CUDA_VISIBLE_DEVICES=0 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine_ln_ --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --no_sn_emb_l
