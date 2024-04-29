from prophet import ProphetInequalityAgent
import numpy as np

class MedianMaxThreshold(ProphetInequalityAgent):
    def __init__(self):
        self.name = 'MedianMaxThreshold'

    def init_new_episode(self, info):
        distrib, num_items = info['distribution'], info['num_items']
        self.num_items = num_items
        self.threshold = distrib.ppf(0.5 ** (1 / num_items))

    def select_action(self, state):
        item_index, item_value = state['item_index'], state['item_value']
        return item_index == self.num_items - 1 or item_value >= self.threshold

class OCRSBased(ProphetInequalityAgent):
    def __init__(self):
        self.name = 'OCRSBased'

    def init_new_episode(self, info):
        distrib, num_items = info['distribution'], info['num_items']
        self.num_items = num_items
        self.threshold = distrib.ppf(1 - 1 / num_items)

    def select_action(self, state):
        item_index, item_value = state['item_index'], state['item_value']
        return item_index == self.num_items - 1 or (item_value >= self.threshold and np.random.binomial(1, 0.5))

class SingleSampleMaxThreshold(ProphetInequalityAgent):
    def __init__(self):
        self.name = 'SingleSampleMaxThreshold'

    def init_new_episode(self, info):
        distrib, num_items = info['distribution'], info['num_items']
        self.num_items = num_items
        self.threshold = np.max(distrib.rvs(num_items))

    def select_action(self, state):
        item_index, item_value = state['item_index'], state['item_value']
        return item_index == self.num_items - 1 or item_value >= self.threshold