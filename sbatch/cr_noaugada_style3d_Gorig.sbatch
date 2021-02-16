#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

srun python ../train_3d.py --outdir=results/noaug_ada_3d_curated_real-Gorig-Dresnet --gpus=4 --cfg=stylegan2-3d-anisobase --data=/data/vision/polina/users/razvan/sungmin/stylegan2/sbatch/script_real_data/TFRecords/TFRecords_Float32_Shuffle0 --kimg=6000 --aug=noaug --snap=1
