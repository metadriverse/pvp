#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {4..7}
do
CUDA_VISIBLE_DEVICES=$i \
nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
--exp_name=cplppo-hgdagger-expert_deter \
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
--remove_loss_1=True \
--remove_loss_3=True \
--remove_loss_6=True \
--training_deterministic=True \
--use_target_policy_only_overwrite_takeover=False \
--use_target_policy=False \
--learning_starts=0 \
--add_bc_loss=True \
--add_bc_loss_only_interventions=True \
--eval_freq=1000 \
--expert_deterministic \
> "0503-exp232233-seed${seeds[$i]}.log" 2>&1 &
done
