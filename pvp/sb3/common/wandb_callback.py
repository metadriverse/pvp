"""
PZH: Copied from official wandb implementation.

W&B callback for sb3

Really simple callback to get logging for each tree

Example usage:

```python
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env


env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
    ),
)
```
"""

import logging
import os

import wandb
from wandb.sdk.lib import telemetry as wb_telemetry

from pvp.sb3.common.callbacks import BaseCallback
from pvp.utils.utils import get_api_key_file

logger = logging.getLogger(__name__)


class WandbCallback(BaseCallback):
    """ Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
    """
    def __init__(
        self,
        trial_name,
        exp_name,
        project_name,
        config=None,
        team_name="",
        verbose: int = 0,
        model_save_path: str = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: str = "all",
    ) -> None:

        # PZH: Setup our key
        WANDB_ENV_VAR = "WANDB_API_KEY"
        key_file_path = get_api_key_file(None)  # Search ~/wandb_api_key_file.txt first, then use PZH's
        with open(key_file_path, "r") as f:
            key = f.readline()
        key = key.replace("\n", "")
        key = key.replace(" ", "")
        os.environ[WANDB_ENV_VAR] = key

        # PZH: A weird bug here and don't know why this fixes
        if "PYTHONUTF8" in os.environ and os.environ["PYTHONUTF8"] == 'on':
            os.environ["PYTHONUTF8"] = '1'

        self.run = wandb.init(
            # Names
            project=project_name,
            id=trial_name,
            group=exp_name,
            entity=team_name,
            # id=exp_name,
            # name=run_name,
            config=config or {},
            resume=True,
            reinit=True,
            sync_tensorboard=True,  # Open this and setup tb in sb3 so that we can get log!
            save_code=True
        )

        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path
        self.gradient_save_freq = gradient_save_freq
        if log not in ["gradients", "parameters", "all", None]:
            wandb.termwarn("`log` must be one of `None`, 'gradients', 'parameters', or 'all', " "falling back to 'all'")
            log = "all"
        self.log = log
        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
            self.path = os.path.join(self.model_save_path, "model.zip")
        else:
            assert (
                self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            wandb.watch(
                self.model.policy,
                log_freq=self.gradient_save_freq,
                log=self.log,
            )
        wandb.config.setdefaults(d)

    def _on_step(self) -> bool:
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    self.save_model()
        return True

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        self.model.save(self.path)
        wandb.save(self.path, base_path=self.model_save_path)
        if self.verbose > 1:
            logger.info(f"Saving model checkpoint to {self.path}")
