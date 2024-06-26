from prophet import ProphetInequalityEnv, ProphetInequalityAgent

from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque
from scipy import stats
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import random

class DQN(ProphetInequalityAgent):
    def __init__(self, env, name):
        super().__init__(env, name)

        self.memory  = deque(maxlen=2000)
        
        self.gamma = 1 # 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def select_action(self, state, training=False):
        state = state.reshape(1, -1)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state, verbose=False)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            state = state.reshape(1, -1)
            new_state = new_state.reshape(1, -1)
            target = self.target_model.predict(state, verbose=False)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state, verbose=False)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def train_one_episode(self):
        cur_state = self.env.reset()
        while True:
            action = self.select_action(cur_state, training=True)
            new_state, reward, done, _ = self.env.step(action)

            # reward = reward if not done else -20
            self.remember(cur_state, action, reward, new_state, done)
            
            self.replay()       # internally iterates default (prediction) model
            self.target_train() # iterates target model

            cur_state = new_state
            if done:
                break

# distribution = stats.uniform(loc=0, scale=100)
# num_items = 15
# env = ProphetInequalityEnv(distribution=distribution, num_items=num_items)

# gamma   = 1 # 0.9
# epsilon = .95

# trials  = 100
# trial_len = 15

# rewards = []
# dqn_agent = DQN(env=env)

# for i in range(trials):
#     avg_reward = 0
#     for trial in range(trials):
#         cur_state = env.reset().reshape(1,-1)
#         for step in range(trial_len):
#             action = dqn_agent.act(cur_state)
#             new_state, reward, done, _ = env.step(action)

#             # reward = reward if not done else -20
#             new_state = new_state.reshape(1,-1)
#             dqn_agent.remember(cur_state, action, reward, new_state, done)
            
#             dqn_agent.replay()       # internally iterates default (prediction) model
#             dqn_agent.target_train() # iterates target model

#             cur_state = new_state
#             if done:
#                 avg_reward += reward
#                 print(f'trial {i} {trial} done')
#                 break
#     rewards.append(avg_reward/trials)

# dqn_agent.save_model("trained.keras")


# newrewards = np.asarray(rewards)
# np.save('rewards',newrewards)

# rewards = np.load('rewards.npy')
# plt.plot(rewards)
# plt.savefig('dqnrewards.svg')