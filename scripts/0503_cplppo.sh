#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {3..4}
do
CUDA_VISIBLE_DEVICES=$i \
nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
--exp_name=cplppo-hgdagger-loss3 \
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
--remove_loss_3=False \
--remove_loss_6=True \
--training_deterministic=True \
--use_target_policy_only_overwrite_takeover=False \
--use_target_policy=False \
--learning_starts=0 \
--add_bc_loss=True \
--add_bc_loss_only_interventions=True \
> "0503-exp2333-seed${seeds[$i]}.log" 2>&1 &
done




# Loop over each GPU
for i in {5..6}
do
CUDA_VISIBLE_DEVICES=$i \
nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
--exp_name=cplppo-hgdagger-loss3-loss6 \
--wandb \
--wandb_project=pvp2024 \
--wandb_team=drivingforce \
--seed=${seeds[$i]} \
--free_level=0.95 \
--use_chunk_adv=True \
--num_steps_per_chunk=64 \
--prioritized_buffer=True \
--learning_starts=0 \
--mask_same_actions=False \
--cpl_bias=0.5 \
--num_comparisons=-1 \
\
--training_deterministic=True \
\
--add_loss_5=False \
--add_loss_5_inverse=False \
\
--remove_loss_1=True \
--remove_loss_3=False \
--remove_loss_6=False \
\
--use_target_policy_only_overwrite_takeover=False \
--use_target_policy=False \
\
--add_bc_loss=True \
--add_bc_loss_only_interventions=True \
\
> "0503-exp2333-seed${seeds[$i]}.log" 2>&1 &
done




# Loop over each GPU
for i in {7..7}
do
CUDA_VISIBLE_DEVICES=$i \
nohup python pvp/experiments/metadrive/train_td3cpl_metadrive_fakehuman.py \
--exp_name=cplppo-hgdagger-loss3-loss5 \
--wandb \
--wandb_project=pvp2024 \
--wandb_team=drivingforce \
--seed=${seeds[$i]} \
--free_level=0.95 \
--use_chunk_adv=True \
--num_steps_per_chunk=64 \
--prioritized_buffer=True \
--learning_starts=0 \
--mask_same_actions=False \
--cpl_bias=0.5 \
--num_comparisons=-1 \
\
--training_deterministic=True \
\
--add_loss_5=True \
--add_loss_5_inverse=True \
\
--remove_loss_1=True \
--remove_loss_3=False \
--remove_loss_6=True \
\
--use_target_policy_only_overwrite_takeover=False \
--use_target_policy=False \
\
--add_bc_loss=True \
--add_bc_loss_only_interventions=True \
\
> "0503-exp2333-seed${seeds[$i]}.log" 2>&1 &
done

