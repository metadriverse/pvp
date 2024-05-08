#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {3..5}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvpes_metadrive_fakehuman.py \
    --exp_name=pvpes-metadrive-regression-remove_negative \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --no_done_for_positive=False \
    --reward_0_for_positive=False \
    --reward_n2_for_intervention=False \
    --reward_1_for_all=False \
    --reward_0_for_negative=False \
    --use_weighted_reward=False \
    --remove_negative=True \
    --adaptive_batch_size=True \
    --agent_data_ratio=1.0 \
    > "0501-agent_data_ratio=1.0-seed${seeds[$i]}.log" 2>&1 &
done

