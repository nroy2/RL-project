import gym
from gym import Env
from gym.spaces import Discrete, Dict, Box
import numpy as np

class ProphetInequalityEnv(Env):
    def __init__(self, distribution, num_items):
        super(ProphetInequalityEnv, self).__init__()

        # Number of total items and the distribution the value will be drawn from
        self.distribution = distribution
        self.num_items = num_items

        # State - defined as tuple of <current item index, current item value>
        self.item_index = 0
        self.item_value = self.distribution.rvs()

        # State space
        self.observation_space = Dict({
            "item_index": Discrete(self.num_items + 1), # account for beyond the last item
            "item_value": Box(low=self.distribution.a, high=self.distribution.b, shape=())
        })

        # Actions we can take: pass on reward, or take reward
        self.action_space = Discrete(2)

        # Reward range
        self.reward_range = (self.distribution.a, self.distribution.b)

    def _get_obs(self):
        return {"item_index": self.item_index, "item_value": self.item_value}

    def _get_info(self):
        return {"distribution": self.distribution, "num_items": self.num_items}

    def step(self, action):
        if action == 1:
            done = True
            observation = self._get_obs()
            reward = self.item_value
            info = self._get_info()

            return observation, reward, done, info
        else:
            self.item_index = min(self.item_index + 1, self.num_items) # disallowing taking more boxes when it's time
            if self.item_index < self.num_items:
                self.item_value = self.distribution.rvs()
            else:
                self.item_value = 0

            done = self.item_index >= self.num_items
            observation = self._get_obs()
            info = self._get_info()
            reward = 0

            return observation, reward, done, info

    def reset(self):
        self.item_index = 0
        self.item_value = self.distribution.rvs()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self, mode='human'):
        pass
