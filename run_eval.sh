CUDA_VISIBLE_DEVICES=0,1 python eval.py \
  --model_path ./saved_models/20221005_cifar10_resnet50_sogclr-128-2048_bz_256_E100_WR10_lr_1.200_sqrt_wd_1e-06_t_0.1_g_0.9_lars_1/stage2_cifar10_cel-128-2048_bz_256_E10_lr_0.005_sqrt_wd_1e-06_t_0.1_lars/model_best.pth.tar \
  --crop-min=.08 \
  --gpu 0 \
  --data_name cifar10 \
  --data ../data/cifar10/ \
  --save_dir ./saved_models/20221005_cifar10_resnet50_sogclr-128-2048_bz_256_E100_WR10_lr_1.200_sqrt_wd_1e-06_t_0.1_g_0.9_lars_1/stage2_cifar10_cel-128-2048_bz_256_E10_lr_0.005_sqrt_wd_1e-06_t_0.1_lars/ \
  --print-freq 10