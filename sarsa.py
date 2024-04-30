import numpy as np
from prophet import ProphetInequalityAgent

class SarsaLambdaAgent(ProphetInequalityAgent):
    def __init__(self, env, name, gamma, lam, alpha, X):
        super().__init__(env, name)
        self.gamma, self.lam, self.alpha, self.X = gamma, lam, alpha, X
        self.total_episode_count = 0
        self.w = np.zeros((X.feature_vector_len()))

    # epsilon greedy
    def select_action(self, state, training=False):
        nA = self.env.action_space.n
        Q = [np.dot(self.w, self.X(state, action)) for action in range(nA)]
        eps = 0.1 if training else 0
        return np.random.randint(nA) if np.random.rand() < eps else np.argmax(Q)

    def train_one_episode(self):
        s, done = self.env.reset(), False
        a = self.select_action(s, training=True)
        x = self.X(s, a)
        z = np.zeros((self.X.feature_vector_len()))
        q_old = 0
        while not done:
            s, r, done, _ = self.env.step(a)
            a_prime = self.select_action(s, training=True)
            x_prime = self.X(s, a_prime) if not done else np.zeros((self.X.feature_vector_len()))
            q, q_prime = self.w @ x, self.w @ x_prime
            delta = r + self.gamma * q_prime - q
            z = self.gamma * self.lam * z + (1 - self.alpha * self.gamma * self.lam * (z @ x)) * x
            self.w = self.w + self.alpha * (delta + q - q_old) * z - self.alpha * (q - q_old) * z
            q_old, x, a = q_prime, x_prime, a_prime
        self.total_episode_count += 1

    def save_model(self, fn):
        with open(fn, 'wb') as f:
            np.save(f, self.w)

    def load_model(self, fn):
        with open(fn, 'rb') as f:
            self.w = np.load(f)

