#!/bin/bash
#SBATCH --gpus=4
#SBATCH --output=./log/train-%j.out
#SBATCH --error=./log/train-%j.out

# module load cudnn/8.9.5.29_cuda12.x
# module load compilers/cuda/12.1
# module load compilers/gcc/12.2.0

# source activate py310cu121_new

# wandb online

accelerate launch --config_file config_bf16.yaml ./train.py \
    --mode bprna \
    --gradient_accumulation_steps 1 \
    --batch_size 1 \
    --lr 1e-4 \
    --select 0.1 --replace 0.3 \
    --pretrained_lm_dir ./model \
    --data_dir ./data \
    --save True

