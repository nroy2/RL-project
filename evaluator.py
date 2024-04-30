from prophet import ProphetInequalityEnv, ProphetInequalityAgent

def evaluate_agent(
    env: ProphetInequalityEnv,
    agent: ProphetInequalityAgent,
    samples_per_eval: int
) -> float:
    """
    Evaluate the average reward of the agent on num_episodes
    """
    total_reward = 0.
    for sample in range(samples_per_eval):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
            state = next_state
    return total_reward / samples_per_eval