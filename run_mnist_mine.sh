CUDA_VISIBLE_DEVICES=1 python mine.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine_ls_cs --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 2 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy

CUDA_VISIBLE_DEVICES=1 python mine.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine_ls_cn --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 2 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy --no_sn_emb_c

CUDA_VISIBLE_DEVICES=1 python mine.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine_ln_cn --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 2 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy --no_sn_emb_c --no_sn_emb_l

CUDA_VISIBLE_DEVICES=1 python mine.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine_ln_cs --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 2 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --use_cy --no_sn_emb_l

CUDA_VISIBLE_DEVICES=1 python mine.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine_ls --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 2 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32

CUDA_VISIBLE_DEVICES=1 python mine.py --dataset mnist --dataroot datasets/mnist --cuda --batchSize 256 --niter 200 --outf results_sn/mnist/mine_ln --use_shared_T --netD_model proj32 --loss_type mine --visualize_class_label 2 --adaptive_grad cd --adaptive --lambda_mi 0.1 --netT_model proj32 --no_sn_emb_l