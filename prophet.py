import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class ProphetInequalityEnv(Env):
    def __init__(self, num_items=100):
        super(ProphetInequalityEnv, self).__init__()
        # Number of total items
        self.num_items = num_items
        
        self.values = [] # TODO: define at some point
        
        self.observation_space = Box() # TODO: fill in with state space once we decide
        
        # Actions we can take: pass on reward, or take reward
        self.action_space = Discrete(2)
        
        self.current_item = 0
        
    def step(self, action):
        
        reward = self._calculate_reward(action)
        
        self.current_item += 1
        
        done = (action == 1) or (self.current_item >= self.num_items)
        
        return self.current_item, reward, done, {}
    
    def _calculate_reward(self, action):
        if (self.current_item >= self.num_items-1) or (action == 1):
            return self.values[self.current_item]
        else:
            return 0
    
    def reset(self):
        self.current_item = 0
        
    def render(self, mode='human'):
        pass
        