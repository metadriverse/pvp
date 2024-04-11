#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)



# Loop over each GPU
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
    --exp_name=pvp-metadrive-fake \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --intervention_start_stop_td=False \
    > "0409_pvp_egpo_freelevel0.95_seed${seeds[$i]}.log" 2>&1 &
done

