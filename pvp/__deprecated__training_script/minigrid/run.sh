#!/usr/bin/env bash

expname="minigrid_dqn_formal"

for ((i = 0; i < 5000; i += 1000)); do
  nohup python train_minigrid_dqn.py \
    --wandb \
    --seed $i \
    --exp-name ${expname} >${expname}_seed${i}.log 2>&1 &
done
