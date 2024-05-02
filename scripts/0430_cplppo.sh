#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {0..2}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.0-add_loss_5=False-0501night-fixed \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.0 \
    --num_comparisons=-1 \
    --add_loss_5=False \
    --prioritized_buffer=True \
    > "0501-exp1-seed${seeds[$i]}.log" 2>&1 &
done


# Loop over each GPU
for i in {2..4}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.25-add_loss_5=False-0501night-fixed \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.25 \
    --num_comparisons=-1 \
    --add_loss_5=False \
    --prioritized_buffer=True \
    > "0501-exp2-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {4..6}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-add_loss_5=False-0501night-fixed \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.5 \
    --num_comparisons=-1 \
    --add_loss_5=False \
    --prioritized_buffer=True \
    > "0501-exp3-seed${seeds[$i]}.log" 2>&1 &
done

