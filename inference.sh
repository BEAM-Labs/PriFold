#!/bin/bash
#SBATCH --gpus=1
#SBATCH --output=./log/%j.out
#SBATCH --error=./log/%j.out

# module load cudnn/8.9.5.29_cuda12.x
# module load compilers/cuda/12.1
# module load compilers/gcc/12.2.0

# source activate py310cu121_new

python inference.py --mode bprna-test --model_scale lx \
                --batch_size 1 \
                --scale 0.01 \
                --model_path ./model/ss_model_bprna.pth \
                --pretrained_lm_dir ./model \
                --data_dir ./data

python inference.py --mode rnastralign-test --model_scale lx \
                --batch_size 1 \
                --scale 0.01 \
                --model_path ./model/ss_model_rnastralign.pth \
                --pretrained_lm_dir ./model \
                --data_dir ./data

python inference.py --mode archiveii-test --model_scale lx \
                --batch_size 1 \
                --scale 0.01 \
                --model_path ./model/ss_model_rnastralign.pth \
                --pretrained_lm_dir ./model \
                --data_dir ./data