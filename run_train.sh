#!/bin/bash
#SBATCH --gpus=4
#SBATCH --output=./log/mars_formerbias-%j.out
#SBATCH --error=./log/mars_formerbias-%j.out

accelerate launch --config_file config_bf16.yaml ./llama2_ss_former_bias_online_outfirst.py \
    --mode bprna \
    --warmup_epos 15 \
    --gradient_accumulation_steps 1 \
    --batch_size 1 \
    --lr 1e-4 \
    --select 0.1 --replace 0.3

