from prophet import ProphetInequalityEnv, ProphetInequalityAgent
import torch

class REINFORCEAgent(ProphetInequalityAgent):
    def __init__(self, env, name, gamma, pi, V):
        super().__init__(env, name)
        self.gamma, self.pi, self.V = gamma, pi, V
    
    def select_action(self, state):
        return self.pi(state)

    def train_one_episode(self):
        S, A, R = [self.env.reset()], [], [0.]
        while True:
            a = self.pi(S[-1])
            s, r, done, _ = self.env.step(a)
            A.append(a)
            S.append(s)
            R.append(r)
            if done:
                break

        T, gamma_t = len(A), 1.

        for t in range(T):
            G, pw = 0., 1.
            for k in range(t + 1, T):
                G += pw * R[k]
                pw *= self.gamma
            delta = G - self.V(S[t])
            self.V.update(S[t], delta)
            self.pi.update(S[t], A[t], gamma_t, delta)
            gamma_t *= self.gamma

    def save_model(self, fn):
        torch.save({
            'V': self.V.state_dict(),
            'pi': self.pi.state_dict()
        }, fn)
    
    def load_model(self, fn):
        checkpoint = torch.load(fn)
        self.V.load_state_dict(checkpoint['V'])
        self.pi.load_state_dict(checkpoint['pi'])