# Proxy Value Propagation (PVP)

<h3>***NeurIPS 2023 Spotlight***</h3>

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

### Metadrive
[Metadrive](https://github.com/metadriverse/metadrive) provides options for three control devices: steering wheel, gamepad and keyboard.

During experiments human subject can always press `E` to pause the experiment and press `Esc` to exit the experiment. The main experiment will run for 40K steps and takes about one hour.

**Steering Wheel (Logitech G29)**
```bash
cd ~/pvp  # Go to the repo root.
python pvp/training_script/metadrive/train_pvp_td3_metadrive.py --control joystick
```
| Action             | Control                 |
|--------------------|-------------------------|
| Throttle           | Throttle pedal          |
| Break              | Break pedal             |
| Human intervention | Left/Right gear shifter |
| Steering           | Steering wheel          |

#### Gamepad (Xbox Wireless Controller)
```bash
python -m training_script.metadrive.train_pvp_td3_metadrive.py --control xboxController
```
| Action             | Control       |
|--------------------|---------------|
| Throttle           | Right trigger |
| Break              | Left trigger  |
| Human intervention | X/A           |
| Steering           | Left stick    |
#### Keyboard
```bash
python -m training_script.metadrive.train_pvp_td3_metadrive.py --control keyboard
```
| Action             | Control                            |
|--------------------|------------------------------------|
| Throttle           | W                                  |
| Break              | S                                  |
| Human intervention | Always intervened when key pressed |
| Steering           | A/D                                |
### Atari games
#### Default
Run default game Breakout-ram-v0: ram observation space
```bash
python -m training_script.atari.train_atari_pvp.py 
```
#### Enable wandb recording 
Please change key file in stable_baseline3/common/wandb_callback.py
```bash
python -m training_script.atari.train_atari_pvp.py --wandb
```
#### Custom game selection 
A full list of support games can be found using `gym.envs.registry.env_specs.keys()`. Meaning of suffix of each game (v0 vs v4) can be found [here](https://github.com/openai/gym/issues/1280#issuecomment-999696133)
```bash
python -m training_script.atari.train_atari_pvp.py --env-name $game_name
```
#### Custom seed
```bash
python -m training_script.atari.train_atari_pvp.py --seed $seed_num
```
#### Control
| Action | Control                            |
|--------|------------------------------------|
| Move   | AWSD                               |
| Fire   | I                                  |


### CARLA Experiment
#### CARLA Installment
Install all necessary dependencies for CARLA from [CARLA official repository](https://github.com/carla-simulator/carla) to install it.
To start training, launch CARLA client by
```bash
./CarlaUE4.sh -carla-rpc-port=9000
```
#### Script
```bash
python train_pvp_carla.py
```
#### Control (Logitech G29)
| Action             | Control                 |
|--------------------|-------------------------|
| Throttle           | Throttle pedal          |
| Break              | Break pedal             |
| Human intervention | Left/Right gear shifter |
| Steering           | Steering wheel          |

### Minigrid Experimennt
#### Minigrid Installment
Please make sure the vestion of gym-minigrid==1.0.3
#### Script
```bash
python -m training_script.minigrid.train_minigrid_pvp.py
```
#### Control (Keyboard)
| Action             | Control           |
|--------------------|-------------------|
| Turn left          | Left button       |
| Turn right         | Right button      |
| Gown Straight      | Up button         |
| Follow agent action| Space/down button |
| Open door / Toggle | "t"               | 
> [!NOTE]
> We are finalizing the code release. Please check out later!

