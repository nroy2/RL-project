from prophet import ProphetInequalityEnv, ProphetInequalityAgent
import numpy as np

class MedianMaxThreshold(ProphetInequalityAgent):
    def __init__(self, env, name):
        super().__init__(env, name)
        info = env._get_info()
        distrib, num_items = info['distribution'], info['num_items']
        self.num_items = num_items
        self.threshold = distrib.ppf(0.5 ** (1 / num_items))

    def select_action(self, state):
        item_index, item_value = state
        return item_index == self.num_items - 1 or item_value >= self.threshold

class OCRSBased(ProphetInequalityAgent):
    def __init__(self, env, name):
        super().__init__(env, name)
        info = env._get_info()
        distrib, num_items = info['distribution'], info['num_items']
        self.num_items = num_items
        self.threshold = distrib.ppf(1 - 1 / num_items)

    def select_action(self, state):
        item_index, item_value = state
        return item_index == self.num_items - 1 or (item_value >= self.threshold and np.random.binomial(1, 0.5))

class SingleSampleMaxThreshold(ProphetInequalityAgent):
    def __init__(self, env, name):
        super().__init__(env, name)
        info = env._get_info()
        distrib, num_items = info['distribution'], info['num_items']
        self.distrib = distrib
        self.num_items = num_items

    def select_action(self, state):
        item_index, item_value = state
        if item_index == 0:
            self.threshold = np.max(self.distrib.rvs(self.num_items))
        return item_index == self.num_items - 1 or item_value >= self.threshold

class OptimalAgent(ProphetInequalityAgent):
    def __init__(self, env, name):
        super().__init__(env, name)
        info = env._get_info()
        distrib, num_items = info['distribution'], info['num_items']
        self.threshold = np.zeros((num_items))
        
        last = 0.
        for i in range(num_items - 1, -1, -1):
            self.threshold[i] = last
            last = distrib.expect(lambda x : max(x, last))

    def select_action(self, state):
        item_index, item_value = state
        return item_value > self.threshold[int(item_index)]

class RandomChoice(ProphetInequalityAgent):
    def __init__(self, env, name):
        super().__init__(env, name)

    def select_action(self, state):
        return 1
