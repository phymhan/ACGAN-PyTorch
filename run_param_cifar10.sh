# ada, noada
# 0.1, 0.5, 1.0, 2.0
# ln_cnz, ls_cs

# mine, mine2

#CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_noa0.1 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 0.1
#CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_noa0.5 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 0.5
#CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_noa1.0 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 1.0
#CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_noa2.0 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 2.0
CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_noa0.2 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 0.2
CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_noa0.02 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 0.02


CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_ada0.1 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 0.1 --adaptive --adaptive_grad cd
CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_ada0.5 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 0.5 --adaptive --adaptive_grad cd
#CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_ada1.0 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 1.0 --adaptive --adaptive_grad cd
#CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_ada2.0 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 2.0 --adaptive --adaptive_grad cd
CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_ada0.2 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 0.2 --adaptive --adaptive_grad cd
CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ln_cnz_ada0.02 --emb_init_zero --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l --lambda_mi 0.02 --adaptive --adaptive_grad cd


CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ls_cs_noa0.1 --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --lambda_mi 0.1

CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ls_cs_noa0.5 --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --lambda_mi 0.5

CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ls_cs_noa0.2 --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --lambda_mi 0.2

CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ls_cs_noa0.02 --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --lambda_mi 0.02


CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ls_cs_ada0.1 --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --adaptive --adaptive_grad cd --lambda_mi 0.1

CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ls_cs_ada0.5 --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --adaptive --adaptive_grad cd --lambda_mi 0.5

CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ls_cs_ada0.2 --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --adaptive --adaptive_grad cd --lambda_mi 0.2

CUDA_VISIBLE_DEVICES=2 python mine.py --dataset cifar10 --dataroot datasets/cifar10 --cuda --batchSize 256 --niter 100 --outf results_new/cifar10/mine_ls_cs_ada0.02 --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 9 --netT_model proj32 --use_cy --adaptive --adaptive_grad cd --lambda_mi 0.02
