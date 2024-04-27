import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class ProphetInequalityEnv(Env):
    def __init__(self, values):
        super(ProphetInequalityEnv, self).__init__()

        # Number of total items and values of each item in order that they will be presented
        self.values = values
        self.num_items = len(values)

        self.current_item = 0
        self.total_reward = 0
        self.state = np.zeros(3)
        self.state[1] = self.num_items
        self.state[2] = self.values[self.current_item]
        
        # State space - defined as tuple of <average rewards seen, number of items left, value of current item>
        self.observation_space = Box(low=0, high=np.inf, shape=(3,), dtype=np.float32) 
        
        # Actions we can take: pass on reward, or take reward
        self.action_space = Discrete(2)
        
    def step(self, action):
        
        reward = self._calculate_reward(action)

        if action==1:
            return self.state, reward, True, {}

        self.total_reward += self.values[self.current_item]
        self.state[0] = self.total_reward / (self.current_item + 1)
        self.state[1] = self.num_items - (self.current_item+1)
        self.state[2] = self.values[self.current_item]
        self.current_item += 1
        
        done = (self.current_item >= self.num_items)
        
        return self.current_item, reward, done, {}
    
    def _calculate_reward(self, action):
        reward = 0

        if (self.current_item >= self.num_items-1) or (action == 1):
            reward =  self.values[self.current_item]

        return reward
    
    def reset(self):
        self.current_item = 0
        self.total_reward = 0
        self.state = np.zeros(2)
        self.state[1] = self.num_items
        self.state[2] = self.values[self.current_item]
        return self.state
        
    def render(self, mode='human'):
        pass
        