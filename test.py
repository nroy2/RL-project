import numpy as np
from scipy import stats
from prophet import ProphetInequalityEnv
from classic_algos import MedianMaxThreshold, OCRSBased, SingleSampleMaxThreshold

# Initialize environment

# any non-negative distribution works
# for example stats.halfnorm() is the |Norm(0, 1)| distribution, stats.uniform(1, 3) is the Unif(1, 4) distribution
distribution = stats.expon() 
num_items = 500
env = ProphetInequalityEnv(distribution=distribution, num_items=num_items)

# Set hyperparameters
num_episodes = 100
max_steps_per_episode = num_items

agents = [MedianMaxThreshold(), OCRSBased(), SingleSampleMaxThreshold()] #TODO: add agents here - for example, q-learning, sarsa, dqn, so on

# Loop over agents
for agent in agents:
    # Loop over episodes
    for episode in range(num_episodes):
        # Reset the environment for each episode
        state, info = env.reset()
        agent.init_new_episode(info)
        
        # Initialize variables for each episode
        total_reward = 0
        
        # Interact with the environment
        for step in range(max_steps_per_episode):

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            
            # If episode is done, break the loop
            if done:
                break
            
            # Update current state
            state = next_state
        
        # Print total reward for each episode
        print(f"Agent: {agent.name}, Episode {episode + 1}, Total Reward: {total_reward}")
