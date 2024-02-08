import numpy as np
from stable_baselines3 import ppo

#imports vector enviornment
from stable_baselines3.common.vec_env import subproc_vec_env

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