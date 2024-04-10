#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)

# Loop over each GPU
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=td3cpl_v1-metadrive-fake \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    > "0409_td3cpl_v1_seed${seeds[$i]}.log" 2>&1 &
done

