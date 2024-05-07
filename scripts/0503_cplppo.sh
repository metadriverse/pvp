#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-loss1_cpl_bias=1.0-bc_loss_weight0.5 \
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
    --mask_same_actions=False \
    --remove_loss_1=False \
    --remove_loss_6=True \
    --training_deterministic=True \
    --use_target_policy_only_overwrite_takeover=False \
    --use_target_policy=False \
    --remove_loss_3=False \
    --bc_loss_weight=0.5 \
    > "0503-exp23-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {4..5}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-loss1_cpl_bias=1.0-bc_loss_weight5 \
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
    --mask_same_actions=False \
    --remove_loss_1=False \
    --remove_loss_6=True \
    --training_deterministic=True \
    --use_target_policy_only_overwrite_takeover=False \
    --use_target_policy=False \
    --remove_loss_3=False \
    --bc_loss_weight=5.0 \
    > "0503-exp23555-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {6..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-cpl_bias=0.5-loss1_cpl_bias=1.0-bc_loss_weight10 \
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
    --mask_same_actions=False \
    --remove_loss_1=False \
    --remove_loss_6=True \
    --training_deterministic=True \
    --use_target_policy_only_overwrite_takeover=False \
    --use_target_policy=False \
    --remove_loss_3=False \
    --bc_loss_weight=10.0 \
    > "0503-exp2310-seed${seeds[$i]}.log" 2>&1 &
done
