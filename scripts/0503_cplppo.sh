#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-remove_loss_6-add_loss_5 \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.5 \
    --num_comparisons=-1 \
    --add_loss_5=True \
    --prioritized_buffer=True \
    --mask_same_actions=False \
    --remove_loss_1=False \
    --remove_loss_6=True \
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=False \
    > "0503-exp415-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {2..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-remove_loss_6-add_loss_5_inverse \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.5 \
    --num_comparisons=-1 \
    --add_loss_5=True \
    --add_loss_5_inverse=True \
    --prioritized_buffer=True \
    --mask_same_actions=False \
    --remove_loss_1=False \
    --remove_loss_6=True \
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=False \
    > "0503-exp125-seed${seeds[$i]}.log" 2>&1 &
done

