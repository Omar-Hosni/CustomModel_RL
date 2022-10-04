import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#build an agent to give us the best shower possible
#randomly temperature
#37 and 39 degrees

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60

    def step(self,action):
        self.state += action -1
        self.shower_length -= 1
        
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1
        if self.showe_length <= 0:
            done = True
        else:
            done = False
            
        info = {}
        return self.state, reward, done, info
    
    def render(self):
        pass
    def reset(self):
        self.state = np.array([38+random.randint(-3,3)]).astype(float)
        self.shower_length = 60
        return self.state
        
        
env = ShowerEnv()
env.observation_space.sample()

log_path = os.path.join('Training','Logs')
model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
model.learn()
