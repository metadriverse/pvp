#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {2..4}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-chunkadv-num_steps_per_chunk64-cpl_bias0.5-num_comparisons-1 \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.5 \
    --num_comparisons=-1 \
    > "0424-cplppo-use_chunk_adv-num_steps_per_chunk64-cpl_bias0.5-num_comparisons-1-seed${seeds[$i]}.log" 2>&1 &
done

# Loop over each GPU
for i in {0..2}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
    --exp_name=cplppo-chunkadv-num_steps_per_chunk64-cpl_bias0.0-num_comparisons-1 \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --use_chunk_adv=True \
    --num_steps_per_chunk=64 \
    --cpl_bias=0.0 \
    --num_comparisons=-1 \
    > "0424-cplppo-use_chunk_adv-num_steps_per_chunk64-cpl_bias0.0-num_comparisons-1-seed${seeds[$i]}.log" 2>&1 &
done