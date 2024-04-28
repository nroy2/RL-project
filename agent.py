class ProphetInequalityAgent(object):
    def __init__(self):
        pass

    # This is only restrcited to classic algorithms
    # RL-based should not use this function
    def init_new_episode(self, info):
        pass

    def select_action(self, state) -> int:
        return 1
    
    def update(self, state, action, reward, next_state):
        pass