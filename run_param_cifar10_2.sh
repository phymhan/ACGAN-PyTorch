# ada, noada
# 0.1, 0.5, 1.0, 2.0
# ln_cnz, ls_cs

# mine, mine2


CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ls_cs_noa0.1 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --lambda_r 0.1

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ls_cs_noa0.5 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --lambda_r 0.5

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ls_cs_noa1.0 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --lambda_r 1.0

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ls_cs_noa2.0 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --lambda_r 2.0


CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ls_cs_ada0.1 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --adaptive --lambda_r 0.1

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ls_cs_ada0.5 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --adaptive --lambda_r 0.5

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ls_cs_ada1.0 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --adaptive --lambda_r 1.0

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ls_cs_ada2.0 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --adaptive --lambda_r 2.0



CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ln_cn_noa0.1 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --no_sn_emb_c --no_sn_emb_l --lambda_r 0.1

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ln_cn_noa0.5 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --no_sn_emb_c --no_sn_emb_l --lambda_r 0.5

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ln_cn_noa1.0 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --no_sn_emb_c --no_sn_emb_l --lambda_r 1.0

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ln_cn_noa2.0 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --no_sn_emb_c --no_sn_emb_l --lambda_r 2.0


CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ln_cn_ada0.1 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --no_sn_emb_c --no_sn_emb_l --adaptive --lambda_r 0.1

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ln_cn_ada0.5 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --no_sn_emb_c --no_sn_emb_l --adaptive --lambda_r 0.5

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ln_cn_ada1.0 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --no_sn_emb_c --no_sn_emb_l --adaptive --lambda_r 1.0

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine2_ln_cn_ada2.0 --use_shared_T --netD_model proj32 --visualize_class_label 9 --netT_model proj32 --use_cy --f_div revkl --loss_type none --no_sn_emb_c --no_sn_emb_l --adaptive --lambda_r 2.0
