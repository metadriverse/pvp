#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)




# Loop over each GPU
for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=pbrl-cpl_bias=1.0-fixedstd-allloss \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=1.0 \
    --num_comparisons=-1 \
    --add_loss_5=False \
    --prioritized_buffer=True \
    --mask_same_actions=False \
    --remove_loss_1=False \
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=False \
    --real_td3 \
    > "0503-exp20-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {2..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=pbrl-cpl_bias=1.0-fixedstd-remove_loss_1 \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=1.0 \
    --num_comparisons=-1 \
    --add_loss_5=False \
    --prioritized_buffer=True \
    --mask_same_actions=False \
    --remove_loss_1=True \
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=False \
    --real_td3 \
    > "0503-exp201-seed${seeds[$i]}.log" 2>&1 &
done


# Loop over each GPU
for i in {4..5}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=pbrl-cpl_bias=0.0-fixedstd-allloss \
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
    --mask_same_actions=False \
    --remove_loss_1=False \
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=False \
    --real_td3 \
    > "0503-exp3-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {6..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=pbrl-cpl_bias=0.5-fixedstd-allloss \
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
    --training_deterministic=True \
    --use_target_policy=False \
    --remove_loss_3=False \
    --real_td3 \
    > "0503-exp4-seed${seeds[$i]}.log" 2>&1 &
done
