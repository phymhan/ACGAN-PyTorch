CUDA_VISIBLE_DEVICES=2 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine2_revkl_ls_cs --use_shared_T --netD_model proj32 --visualize_class_label 9 --adaptive --lambda_r 0.1 --netT_model proj32 --use_cy --f_div revkl

CUDA_VISIBLE_DEVICES=2 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine2_revkl_ls_cn --use_shared_T --netD_model proj32 --visualize_class_label 9 --adaptive --lambda_r 0.1 --netT_model proj32 --use_cy --f_div revkl --no_sn_emb_c

CUDA_VISIBLE_DEVICES=2 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine2_revkl_ln_cn --use_shared_T --netD_model proj32 --visualize_class_label 9 --adaptive --lambda_r 0.1 --netT_model proj32 --use_cy --f_div revkl --no_sn_emb_c --no_sn_emb_l

CUDA_VISIBLE_DEVICES=2 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine2_revkl_ln_cs --use_shared_T --netD_model proj32 --visualize_class_label 9 --adaptive --lambda_r 0.1 --netT_model proj32 --use_cy --f_div revkl --no_sn_emb_l

CUDA_VISIBLE_DEVICES=2 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine2_revkl_ln --use_shared_T --netD_model proj32 --visualize_class_label 9 --adaptive --lambda_r 0.1 --netT_model proj32 --f_div revkl --no_sn_emb_l

CUDA_VISIBLE_DEVICES=2 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 200 --outf results_sn/cifar10/mine2_revkl_ls --use_shared_T --netD_model proj32 --visualize_class_label 9 --adaptive --lambda_r 0.1 --netT_model proj32 --f_div revkl