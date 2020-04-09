CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine2_revkl_ls_cs --use_shared_T --netD_model proj32 --visualize_class_label 2 --netT_model proj32 --use_cy --f_div revkl --adaptive --lambda_r 0.1

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine2_revkl_ls_cn --use_shared_T --netD_model proj32 --visualize_class_label 2 --netT_model proj32 --use_cy --f_div revkl --adaptive --lambda_r 0.1 --no_sn_emb_c

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine2_revkl_ln_cn --use_shared_T --netD_model proj32 --visualize_class_label 2 --netT_model proj32 --use_cy --f_div revkl --adaptive --lambda_r 0.1 --no_sn_emb_c --no_sn_emb_l

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine2_revkl_ln_cs --use_shared_T --netD_model proj32 --visualize_class_label 2 --netT_model proj32 --use_cy --f_div revkl --adaptive --lambda_r 0.1 --no_sn_emb_l

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine2_revkl_ln --use_shared_T --netD_model proj32 --visualize_class_label 2 --netT_model proj32 --f_div revkl --adaptive --lambda_r 0.1 --no_sn_emb_l

CUDA_VISIBLE_DEVICES=3 python mine2.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine2_revkl_ls --use_shared_T --netD_model proj32 --visualize_class_label 2 --netT_model proj32 --f_div revkl --adaptive --lambda_r 0.1