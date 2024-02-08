import numpy as np
from stable_baselines3 import PPO

#imports vector enviornment
from stable_baselines3.common.vec_env import SubprocVecEnv

#This plots the results
from stable_baselines3.common.results_plotter import load_results, ts2xy
#Helps make it so even with randome environments we get results
from stable_baselines3.common.utils import set_random_seed

#imports base call back
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
#Skips frames to learn a bit better
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

import os
import retro

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

#adds a new environment
def make_env(env_id, rank, seed =0):
    def _init():
        env = retro.make(game = env_id)
        #Makes a decision every four frames
        env = MaxAndSkipEnv(env,4)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init
#log directory
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok = True)
#four cpus with this game
env_id = "Airstriker-Genesis"
num_cpu = 4
#runs four environments at once
env = VecMonitor(SubprocVecEnv([make_env(env_id,i) for i in range(num_cpu)]), "tmp/monitor")
#Our model is going to be a PPO model
#Verbose = 1. How  much it tells us as it trains.
#We change our learning_rate
model = PPO('CnnPolicy', env, verbose= 1, tensorboard_log ="./board/", learning_rate=0.00003)
#If you have a model already loaded/made then you can do this
#model = PPO.load("path to the model", env = env)

print("------START LEARNING YAYYY------")

#callback saves the best model each time
#Every 1000 steps the best model is saved

#Adds a callback place. Basically picks the best of the 1000 models in log_dir
callback = SaveOnBestTrainingRewardCallback(chef_freq=1000,log_dir = log_dir)
#tells the model how to learn
model.learn(total_timesteps = 1000000, callback = callback, tb_log_name = "first try!!!")