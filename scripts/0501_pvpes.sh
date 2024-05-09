#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvpes_metadrive_fakehuman.py \
    --exp_name=pvpes-metadrive-expert_sto-add_bc_loss \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --adaptive_batch_size=True \
    --agent_data_ratio=1.0 \
    --add_bc_loss=True \
    --eval_freq=1000 \
    > "0501-agent_data_ratio=1.0-seed${seeds[$i]}.log" 2>&1 &
done


# Loop over each GPU
for i in {2..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvpes_metadrive_fakehuman.py \
    --exp_name=pvpes-metadrive-regression-expert_deterministic \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --adaptive_batch_size=True \
    --agent_data_ratio=1.0 \
    --add_bc_loss=False \
    --eval_freq=1000 \
    --expert_deterministic \
    > "0501-agent_data_ratio=1.0-seed${seeds[$i]}.log" 2>&1 &
done
