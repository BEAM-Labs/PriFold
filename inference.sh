#!/bin/bash
#SBATCH --gpus=1
#SBATCH --output=./log/%j.out
#SBATCH --error=./log/%j.out

python inference.py --mode bprna --model_scale lx \
                --batch_size 1 \
                --scale 0.01 \
                --select 0.1 --replace 0.4 \
                --model_path ./model/ss_model_bprna.pth \
                --pretrained_lm_dir ./model \
                --data_dir ../data