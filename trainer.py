from prophet import ProphetInequalityEnv, ProphetInequalityAgent
from evaluator import evaluate_agent
import numpy as np
from tqdm import tqdm

def train_agent(
    env: ProphetInequalityEnv,
    agent: ProphetInequalityAgent,
    num_episodes: int,
    evaluate_interval: int,
    samples_per_eval: int
) -> np.array:
    """
    Train agent on num_episodes, return the average reward of agent every fixed interval
    """
    rewards = []
    pbar = tqdm(range(num_episodes), desc=f"Agent: {agent.name}, Episode 0")
    for episode in tqdm(range(num_episodes)):
        agent.train_one_episode()
        if (episode + 1) % evaluate_interval == 0:
            reward = evaluate_agent(env, agent, samples_per_eval)
            pbar.set_description(f"Agent: {agent.name}, Episode {episode + 1}, Average Reward: {reward}")
            rewards.append(reward)

    return np.array(rewards)