# Reward-free Policy Learning through Active Human Involvement

[//]: # (Reward-free Policy Learning through Active Human Involvement)

## Installation

```bash
# Clone the code to local machine
git clone XXX
cd PVP

# Create virtual environment
conda create -n pvp python=3.7
conda activate pvp

# Install dependencies
pip install -e .
pip install -r requirements.txt 

# Install evdev package (Linux only)
pip install evdev

# Install dependencies for CARLA experiment (recommended to create another conda environment)
pip install di-engine==0.2.0 markupsafe==2.0.1

# Then install the compatible PyTorch version from https://pytorch.org/
```



## Training script
### Metadrive
For Metadrive we provided three control options for generalizability. During experiments human subject can always press *E* to pause the experiment. This experiment will run for 40K steps and takes about one hour.
#### Steering Wheel (Logitech G29)
```bash
python train_pvp_td3_metadrive.py --control joystick
```
| Action             | Control                 |
|--------------------|-------------------------|
| Throttle           | Throttle pedal          |
| Break              | Break pedal             |
| Human intervention | Left/Right gear shifter |
| Steering           | Steering wheel          |

#### Gamepad (Xbox Wireless Controller)
```bash
python train_pvp_td3_metadrive.py --control xboxController
```
| Action             | Control       |
|--------------------|---------------|
| Throttle           | Right trigger |
| Break              | Left trigger  |
| Human intervention | X/A           |
| Steering           | Left stick    |
#### Keyboard
```bash
python train_pvp_td3_metadrive.py --control keyboard
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
python train_atari_pvp.py 
```
#### Enable wandb recording 
Please change key file in stable_baseline3/common/wandb_callback.py
```bash
python train_atari_pvp.py --wandb
```
#### Custom game selection 
A full list of support games can be found using `gym.envs.registry.env_specs.keys()`. Meaning of suffix of each game (v0 vs v4) can be found [here](https://github.com/openai/gym/issues/1280#issuecomment-999696133)
```bash
python train_atari_pvp.py --env-name $game_name
```
#### Custom seed
```bash
python train_atari_pvp.py --seed $seed_num
```
#### Control
| Action | Control                            |
|--------|------------------------------------|
| Move   | AWSD                               |
| Fire   | I                                  |


### CARLA Experimennt
#### CARLA Installment
Install all necessary dependencies for CARLA from [CARLA offical repository](https://github.com/carla-simulator/carla) to install it.
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
Please only install gym-minigrid==1.0.3 as shown in the requirments.txt, latest gym-minigrid might cause trouble
```bash
pip install gym-minigrid==1.0.3
```

```bash
./CarlaUE4.sh -carla-rpc-port=9000
```
#### Script
```bash
python train_minigrid_pvp.py
```
#### Control (Keyboard)
| Action             | Control                 |
|--------------------|-------------------------|
| Turn left          | Left button             |
| Turn right         | Right button            |
| Gown Straight      | Up button               |
| Follow agent action| Space/down button       |
| Open door / Toggle | "t"
