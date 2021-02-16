#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

srun python ../train_3d.py --outdir=results/noaug_ada_3d_64-Gorig-Dresnet --gpus=4 --cfg=stylegan2-3d --data=/data/vision/polina/users/razvan/sungmin/data/Synthetic/646464_Uniform/Train/TFRecords_Jan2021 --kimg=6000 --aug=noaug --snap=1
