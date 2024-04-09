nohup python pvp/experiments/metadrive/train_cpl_metadrive_fakehuman.py \
--exp_name=pvp-metadrive-fake-free0.95 \
--wandb \
--wandb_project=pvp2024 \
--wandb_team=drivingforce \
--seed=0 \
> 0409_pvp_egpo_seed0.log 2>&1 &

sleep 2

nohup python pvp/experiments/metadrive/train_cpl_metadrive_fakehuman.py \
--exp_name=pvp-metadrive-fake-free0.95 \
--wandb \
--wandb_project=pvp2024 \
--wandb_team=drivingforce \
--seed=100 \
> 0409_pvp_egpo_seed100.log 2>&1 &

sleep 2

nohup python pvp/experiments/metadrive/train_cpl_metadrive_fakehuman.py \
--exp_name=pvp-metadrive-fake-free0.95 \
--wandb \
--wandb_project=pvp2024 \
--wandb_team=drivingforce \
--seed=200 \
> 0409_pvp_egpo_seed200.log 2>&1 &