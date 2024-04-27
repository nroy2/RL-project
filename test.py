import numpy as np
from prophet import ProphetInequalityEnv

# Initialize environment
item_values = [1,2,3,4,5]
env = ProphetInequalityEnv(values=item_values)

# Set hyperparameters
num_episodes = 100
max_steps_per_episode = len(item_values)

agents = [] #TODO: add agents here - for example, q-learning, sarsa, dqn, so on

# Loop over agents
for agent in agents:
    # Loop over episodes
    for episode in range(num_episodes):
        # Reset the environment for each episode
        state = env.reset()
        
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
