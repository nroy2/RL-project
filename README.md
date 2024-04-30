# Prophet Inequalities over I.I.D. Items via Reinforcement Learning

Take a look at our writeup [here](https://www.overleaf.com/read/ktgkygtjbqsr#8d1d9f).

## Supported Models
We support 3 types of RL models. These models are generic (i.e. they don't need to know the distribution). These are:
- `DQN`: Deep Q-Network.
- `SarsaLambdaAgent`: True online Sarsa(λ), implemented with tile coding value approximation. Implementation derived from one author's assignment.
- `REINFORCEAgent`: REINFORCE, implemented with neural network value/pi approximation. Implementation derived from one author's assignment.

We also implement 5 algorithms known from literature. Apart from `RandomChoice`, the remaining algorithms must know the distribution beforehand.

## Training Models
Inside `test.py`, you should see 2 functions:
- `train_and_compare_models`: This function is used to train models. The parameters are:
    - `test_name`: a distinguished name for test.
    - `distribution`: an instance of `scipy.stats.rv_continuous` distribution that is common for all item rewards (e.g. all item rewards are drawn I.I.D. from this distribution).
    - `num_items`: number of items.
    - `num_episodes`: number of episodes to train each model.
    - `evaluate_interval`: evaluate the performance of each model every `evaluate_interval` episodes trained.
    - `samples_per_eval`: the number of episodes to run (to take an average over) every time we evaluate. Note that because the environment itself is probabilistic, we recommend setting this value to at least 100 to eliminate noises inherent of the environment.

To train some models, go inside the function's implementation, put the desired model in the `agents` array, and then run the function outside. You should see a progress bar indicating that a model is training, along with its last evaluation.

## Plotting Previous Rewards
`train_and_compare_models` store evaluation results into the `reward_file` folder. To plot an old test, simply run `plot_test_rewards(test_name, distribution, num_items, num_episodes, evaluate_interval)`, with the parameters being exactly similar as the one put into `train_and_compare_models`. The plot will be stored into the `plot_file` folder with the format `test_name.png`.

## Restoring Models
We have some previously trained model stored in `model_file`. To restore them, you first need to initialize a model with the exact environment and parameter, then simply do `model.load_model(file_name)`. For example, to load `model_file/Expon-15-1m-10k-10k_SarsaLambdaAgent`, use the following snippet:

```py
distribution, num_items = stats.expon(loc=0, scale=100), 15
env = ProphetInequalityEnv(distribution=distribution, num_items=num_items)

# Tile coding for SARSA Agent
reward_low, reward_high = distribution.interval(confidence=0.999)
tile_coding = StateActionFeatureVectorWithTile(
    state_low=np.array([0, reward_low]),
    state_high=np.array([num_items + 1, reward_high]),
    num_actions=2,
    num_tilings=1,
    tile_width=np.array([1, (reward_high - reward_low) / 100])
)

model = SarsaLambdaAgent(env=env, name='SarsaLambdaAgent', gamma=1, lam=0.2, alpha=0.05, X=tile_coding)
model.load_model('model_file/Expon-15-1m-10k-10k_SarsaLambdaAgent')
```
