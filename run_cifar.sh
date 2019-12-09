#!/bin/bash
GPUID=2
dataset_path=~/bj/dataset/
checkpoint_path=~/bj/checkpoint/openset

dataset=cifar10
arch=ae
depth=32

TRAIN=false
EVAL=true


INLIERS=543210
INLIERS=875421
INLIERS=965410
INLIERS=987652
INLIERS=865430
INLIERS=984320


ExpName=${dataset}_${arch}${depth}_$INLIERS

if $TRAIN; then
echo RUN TRAINING
NV_GPU=${GPUID} nvidia-docker run -v `pwd`:`pwd` \
                  -v ${dataset_path}:`pwd`/data/ \
                  -v ${checkpoint_path}:`pwd`/checkpoints/ \
                  -w `pwd` \
                  --rm -it \
                  --ipc=host \
                  --name ${ExpName} \
                  feidfoe/pytorch:v.3 \
                  python cifar.py -a $arch \
                                  --depth $depth \
                                  --dataset $dataset \
                                  --inlier $INLIERS \
                                  --epoch 180 \
                                  --schedule 80 150 \
                                  --gamma 0.1 \
                                  --wd 5e-4 \
                                  --checkpoint checkpoints/${ExpName} 


else
    echo DO NOT RUN TRAINING


fi




if $EVAL; then
echo RUN EVALUATION
CKPT=checkpoints/${ExpName}/model_best.pth.tar
NV_GPU=${GPUID} nvidia-docker run -v `pwd`:`pwd` \
                  -v ${dataset_path}:`pwd`/data/ \
                  -v ${checkpoint_path}:`pwd`/checkpoints/ \
                  -w `pwd` \
                  --rm -it \
                  --ipc=host \
                  feidfoe/pytorch:v.3 \
                  python cifar.py -a $arch \
                                  --depth $depth \
                                  --dataset $dataset \
                                  --inlier $INLIERS \
                                  --checkpoint checkpoints/${ExpName} \
                                  --resume ${CKPT} \
                                  --evaluate
else
    echo DO NOT RUN EVALUATION
fi



