import gym
from gym import Env
from gym.spaces import Discrete, Box, Tuple
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

        # Reward range
        self.reward_range = (self.distribution.ppf(0), self.distribution.ppf(1))

        # State space
        self.observation_space = Box(low=np.array([0, self.reward_range[0]]), high=np.array([num_items, self.reward_range[1]]))
        # self.observation_space = Tuple(Discrete(self.num_items + 1), Box(low=self.reward_range[0], high=self.reward_range[1], shape=(1)))

        # Actions we can take: pass on reward, or take reward
        self.action_space = Discrete(2)

    def _get_obs(self):
        return {"item_index": self.item_index, "item_value": self.item_value}

    def _get_info(self):
        return {"distribution": self.distribution, "num_items": self.num_items, "reward_range": self.reward_range}

    def step(self, action):
        if action == 1:
            done = True
            state = np.array([self.item_index, self.item_value])
            reward = self.item_value
            info = self._get_info()

            return state, reward, done, info
        else:
            self.item_index = min(self.item_index + 1, self.num_items) # disallowing taking more boxes when it's time
            if self.item_index < self.num_items:
                self.item_value = self.distribution.rvs()
            else:
                self.item_value = 0

            done = self.item_index >= self.num_items
            state = np.array([self.item_index, self.item_value])
            info = self._get_info()
            reward = 0

            return state, reward, done, info

    def reset(self):
        self.item_index = 0
        self.item_value = self.distribution.rvs()

        state = np.array([self.item_index, self.item_value])
        return state

    def render(self, mode='human'):
        pass

class ProphetInequalityAgent(object):
    def __init__(self, env: ProphetInequalityEnv, name: str):
        self.env, self.name = env, name

    def select_action(self, state) -> int:
        return 1

    def train_one_episode(self):
        pass
    
    def save_model(self, fn):
        pass

    def load_model(self, fn):
        pass