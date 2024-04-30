import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from trainer import train_agent
from prophet import ProphetInequalityEnv

from tile_coding import StateActionFeatureVectorWithTile
from nn import VApproximationWithNN, PiApproximationWithNN

from classic_algos import *
from sarsa import SarsaLambdaAgent
from reinforce import REINFORCEAgent
from dqn import DQN

def train_and_compare_models(test_name, distribution, num_items, num_episodes, evaluate_interval, samples_per_eval):
    plt.title(test_name)

    env = ProphetInequalityEnv(distribution=distribution, num_items=num_items)

    # Tile coding for SARSA Agent
    reward_low, reward_high = distribution.interval(confidence=0.999)
    print(reward_low, reward_high)
    tile_coding = StateActionFeatureVectorWithTile(
        state_low=np.array([0, reward_low]),
        state_high=np.array([num_items + 1, reward_high]),
        num_actions=2,
        num_tilings=1,
        tile_width=np.array([1, (reward_high - reward_low) / 100])
    )
    # NN pi/V for REINFORCE Agent
    pi = PiApproximationWithNN(2, 2, 3e-4)
    V = VApproximationWithNN(2, 3e-4)

    # Add agents here
    agents = [
        # DQN(env),
        # SarsaLambdaAgent(env=env, gamma=1, lam=0.2, alpha=0.05, X=tile_coding),
        # REINFORCEAgent(env=env, gamma=1, pi=pi, V=V),
        MedianMaxThreshold(env),
        OCRSBased(env),
        SingleSampleMaxThreshold(env),
        RandomChoice(env),
    ]

    for agent in agents:
        rewards = train_agent(env, agent, num_episodes, evaluate_interval, samples_per_eval)
        plt.plot(range(evaluate_interval, num_episodes + 1, evaluate_interval), rewards, label=agent.name)
        agent.save_model(fn=f"model_file/{test_name}_{agent.name}")

    plt.legend()
    plt.savefig(f"plot_file/{test_name}.png", dpi=1000)
    plt.clf()

# Exponential
train_and_compare_models(
    test_name='Expon-15',
    distribution=stats.expon(loc=0, scale=100),
    num_items=15,
    num_episodes=100_000,
    evaluate_interval=1_000,
    samples_per_eval=10_000
)

# Uniform
train_and_compare_models(
    test_name='Uniform-15',
    distribution=stats.uniform(loc=0, scale=100),
    num_items=15,
    num_episodes=100_000,
    evaluate_interval=1_000,
    samples_per_eval=10_000
)

# Half Normal
train_and_compare_models(
    test_name='Halfnorm-15',
    distribution=stats.halfnorm(loc=0, scale=100),
    num_items=15,
    num_episodes=100_000,
    evaluate_interval=1_000,
    samples_per_eval=10_000
)

# # Initialize environment

# # any non-negative distribution works
# # for example stats.halfnorm() is the |Norm(0, 1)| distribution, stats.uniform(1, 3) is the Unif(1, 4) distribution
# distribution = stats.expon(loc=0, scale=100)
# num_items = 15
# env = ProphetInequalityEnv(distribution=distribution, num_items=num_items)

# # Set hyperparameters
# num_episodes = 1_000_000
# evaluate_interval = 10_000
# samples_per_eval = 10_000

# # Plotting stuff

# for agent in agents:
#     rewards = train_agent(env, agent, num_episodes, evaluate_interval, samples_per_eval)
#     plt.plot(range(evaluate_interval, num_episodes + 1, evaluate_interval), rewards, label=agent.name)
# plt.legend()
# plt.show()

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
