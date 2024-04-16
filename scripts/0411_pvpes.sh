#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300)

# Loop over each GPU
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvpes_metadrive_fakehuman.py \
    --exp_name=pvpes-metadrive-no_done_for_positive \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --no_done_for_positive=True \
    --reward_0_for_positive=False \
    --reward_n2_for_intervention=False \
    --reward_1_for_all=False \
    --reward_0_for_negative=False \
    > "0412-pvpes-no_done_for_positive-seed${seeds[$i]}.log" 2>&1 &
done


# Loop over each GPU
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvpes_metadrive_fakehuman.py \
    --exp_name=pvpes-metadrive-no_done_for_positive-reward_0_for_positive \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --no_done_for_positive=True \
    --reward_0_for_positive=True \
    --reward_n2_for_intervention=False \
    --reward_1_for_all=False \
    --reward_0_for_negative=False \
    > "0412-pvpes-reward_0_for_positive-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvpes_metadrive_fakehuman.py \
    --exp_name=pvpes-metadrive-reward_n2_for_intervention \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --no_done_for_positive=False \
    --reward_0_for_positive=False \
    --reward_n2_for_intervention=True \
    --reward_1_for_all=False \
    --reward_0_for_negative=False \
    > "0412-pvpes-reward_n2_for_intervention-seed${seeds[$i]}.log" 2>&1 &
done

# Define the seeds for each GPU
seeds=(0 100 200 300)

# Loop over each GPU
for i in {4..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvpes_metadrive_fakehuman.py \
    --exp_name=pvpes-metadrive-reward_1_for_all \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --no_done_for_positive=False \
    --reward_0_for_positive=False \
    --reward_n2_for_intervention=False \
    --reward_1_for_all=True \
    --reward_0_for_negative=False \
    > "0412-pvpes-reward_1_for_all-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {4..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvpes_metadrive_fakehuman.py \
    --exp_name=pvpes-metadrive-reward_0_for_negative \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --no_done_for_positive=False \
    --reward_0_for_positive=False \
    --reward_n2_for_intervention=False \
    --reward_1_for_all=False \
    --reward_0_for_negative=True \
    > "0412-pvpes-reward_0_for_negative-seed${seeds[$i]}.log" 2>&1 &
done

