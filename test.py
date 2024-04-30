import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from trainer import train_agent
from prophet import ProphetInequalityEnv

from tile_coding import StateActionFeatureVectorWithTile

from classic_algos import *
from sarsa import SarsaLambdaAgent
from dqn import DQN

# Initialize environment

# any non-negative distribution works
# for example stats.halfnorm() is the |Norm(0, 1)| distribution, stats.uniform(1, 3) is the Unif(1, 4) distribution
distribution = stats.uniform(loc=0, scale=100)
num_items = 15
env = ProphetInequalityEnv(distribution=distribution, num_items=num_items)
reward_low, reward_high = env.reward_range

# Set hyperparameters
num_episodes = 100
evaluate_interval = 1
samples_per_eval = 100

# Plotting stuff

# Sarsa agent
tile_coding = StateActionFeatureVectorWithTile(
    state_low=np.array([0, reward_low]),
    state_high=np.array([num_items + 1, reward_high]),
    num_actions=2,
    num_tilings=1,
    tile_width=np.array([1, 1])
)

#TODO: add agents here - for example, q-learning, sarsa, dqn, so on
agents = [
    DQN(env)
    # SarsaLambdaAgent(env=env, gamma=1, lam=0.2, alpha=0.05, X=tile_coding),
    # MedianMaxThreshold(env),
    # OCRSBased(env),
    # SingleSampleMaxThreshold(env),
    # RandomChoice(env)
]

for agent in agents:
    rewards = train_agent(env, agent, num_episodes, evaluate_interval, samples_per_eval)
    plt.plot(range(evaluate_interval, num_episodes + 1, evaluate_interval), rewards, label=agent.name)
plt.legend()
plt.show()

# # Loop over agents
# for agent in agents:
#     # Loop over episodes
#     for episode in range(num_episodes):
#         # Reset the environment for each episode
#         state, info = env.reset()
#         agent.init_new_episode(info)
        
#         # Initialize variables for each episode
#         total_reward = 0
        
#         # Interact with the environment
#         for step in range(max_steps_per_episode):

#             action = agent.select_action(state)
#             next_state, reward, done, _ = env.step(action)
#             agent.update(state, action, reward, next_state)
            
#             total_reward += reward
            
#             # If episode is done, break the loop
#             if done:
#                 break
            
#             # Update current state
#             state = next_state
        
#         # Print total reward for each episode
#         print(f"Agent: {agent.name}, Episode {episode + 1}, Total Reward: {total_reward}")
