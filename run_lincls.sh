CUDA_VISIBLE_DEVICES=0,1 python lincls.py \
  --dist-url 'tcp://localhost:10003' \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 24 \
  --pretrained ./saved_models/20221005_cifar10_resnet50_sogclr-128-2048_bz_256_E350_WR10_lr_1.200_sqrt_wd_1e-06_t_0.1_g_0.9_lars_1/model_best.pth.tar \
  --data_name cifar10 \
  --data ../data/cifar10/ \
  --save_dir ./saved_models/20221005_cifar10_resnet50_sogclr-128-2048_bz_256_E350_WR10_lr_1.200_sqrt_wd_1e-06_t_0.1_g_0.9_lars_1