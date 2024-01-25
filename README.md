# Proxy Value Propagation (PVP)

<h3><b>NeurIPS 2023 Spotlight</b></h3>

Official release for the code used in paper: *Learning from Active Human Involvement through Proxy Value Propagation*

[**Webpage**](https://metadriverse.github.io/pvp/) | 
[**Code**](https://github.com/metadriverse/pvp) |
[**Paper**](https://openreview.net/pdf?id=q8SukwaEBy)



**TODO: A teaser figure here.**


## Installation

```bash
# Clone the code to local machine
git clone https://github.com/metadriverse/pvp
cd pvp

# Create Conda environment
conda create -n pvp python=3.7
conda activate pvp

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install evdev package (Linux only)
pip install evdev

# Install dependencies for CARLA experiment (recommended to create another conda environment)
pip install di-engine==0.2.0 markupsafe==2.0.1

# Then install the compatible PyTorch version from https://pytorch.org/
```



## Launch Experiments

### MetaDrive

[Metadrive](https://github.com/metadriverse/metadrive) provides options for three control devices: steering wheel, gamepad and keyboard.

During experiments human subject can always press `E` to pause the experiment and press `Esc` to exit the experiment. The main experiment will run for 40K steps and takes about one hour. For toy environment with `--toy_env`, it takes about 10 minutes.

Click for the experiment details:



<details>
  <summary><b>MetaDrive - Keyboard</b></summary>

```bash
# Go to the repo root
cd ~/pvp

# Run toy experiment
python pvp/experiments/metadrive/train_pvp_metadrive.py \
--device keyboard \
--toy_env \
--exp_name pvp_metadrive_toy_keyboard

# Run full experiment
python pvp/experiments/metadrive/train_pvp_metadrive.py \
--device keyboard \
--exp_name pvp_metadrive_keyboard \
--wandb \
--wandb_project WADNB_PROJECT_NAME \
--wandb_team WANDB_ENTITY_NAME
```

| Action             | Control       |
|--------------------|---------------|
| Steering           | A/D           |
| Throttle           | W             |
| Human intervention | Space or WASD |
</details>




<details>
  <summary><b>MetaDrive - Steering Wheel (Logitech G29)</b></summary>

Note: Do not connect Xbox controller with the steering wheel at the same time!

```bash
# Go to the repo root
cd ~/pvp

# Run toy experiment
python pvp/experiments/metadrive/train_pvp_metadrive.py \
--device wheel \
--toy_env \
--exp_name pvp_metadrive_toy_wheel

# Run full experiment
python pvp/experiments/metadrive/train_pvp_metadrive.py \
--device wheel \
--exp_name pvp_metadrive_wheel \
--wandb \
--wandb_project WADNB_PROJECT_NAME \
--wandb_team WANDB_ENTITY_NAME
```


| Action             | Control                 |
|--------------------|-------------------------|
| Steering           | Steering wheel          |
| Throttle           | Throttle pedal          |
| Human intervention | Left/Right gear shifter |
</details>



<details>
  <summary><b>MetaDrive - Gamepad (Xbox Wireless Controller)</b></summary>

Note: Do not connect Xbox controller with the steering wheel at the same time!

```bash
# Go to the repo root
cd ~/pvp

# Run toy experiment
python pvp/experiments/metadrive/train_pvp_metadrive.py \
--device gamepad \
--toy_env \
--exp_name pvp_metadrive_toy_gamepad

# Run full experiment
python pvp/experiments/metadrive/train_pvp_metadrive.py \
--device gamepad \
--exp_name pvp_metadrive_gamepad \
--wandb \
--wandb_project WADNB_PROJECT_NAME \
--wandb_team WANDB_ENTITY_NAME
```
| Action             | Control                    |
|--------------------|----------------------------|
| Steering           | Left-right of Left Stick   |
| Throttle           | Up-down of Right Stick     |
| Human intervention | X/A/B & Left/Right Trigger |
</details>


### CARLA

Coming soon!

### Minigrid

Coming soon!



## 📎 References

```latex
@inproceedings{peng2023learning,
  title={Learning from Active Human Involvement through Proxy Value Propagation},
  author={Peng, Zhenghao and Mo, Wenjie and Duan, Chenda and Li, Quanyi and Zhou, Bolei},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

