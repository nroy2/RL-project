from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PiApproximationWithNN(nn.Module):
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        super(PiApproximationWithNN, self).__init__()
        self.num_actions = num_actions
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32).double(),
            nn.ReLU().double(),
            nn.Linear(32, 32).double(),
            nn.ReLU().double(),
            nn.Linear(32, num_actions).double(),
            nn.Softmax().double()
        )
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=alpha,
            betas=[0.9, 0.999]
        )

    def forward(self, states, return_prob=False):
        # TODO: implement this method

        # Note: You will want to return either probabilities or an action
        # Depending on the return_prob parameter
        # This is to make this function compatible with both the
        # update function below (which needs probabilities)
        # and because in test cases we will call pi(state) and 
        # expect an action as output.
        self.model.eval()
        probs = self.model(torch.from_numpy(states))
        if return_prob:
            return probs
        else:
            a = np.random.choice(np.arange(self.num_actions), p=probs.detach().numpy())
            return a

    def update(self, states, actions_taken, gamma_t, delta):
        """
        states: states
        actions_taken: actions_taken
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.model.eval()
        action_prob = self.forward(states, True)
        action_prob = action_prob.gather(0, torch.from_numpy(np.array([actions_taken])))
        loss = torch.mean(-torch.log(action_prob) * delta * gamma_t)
        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    There is no need to change this class.
    """
    def __init__(self,b):
        self.b = b
        
    def __call__(self, states):
        return self.forward(states)
        
    def forward(self, states) -> float:
        return self.b

    def update(self, states, G):
        pass

class VApproximationWithNN(nn.Module):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32).double(),
            nn.ReLU().double(),
            nn.Linear(32, 32).double(),
            nn.ReLU().double(),
            nn.Linear(32, 1).double()
        )
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=alpha,
            betas=[0.9, 0.999]
        )

    def forward(self, states) -> float:
        self.model.eval()
        val = self.model(torch.from_numpy(states))
        return val.detach().numpy()[0]

    def update(self, states, G):
        self.model.eval()
        val = self.model(torch.from_numpy(states))
        loss_fn = nn.MSELoss()
        loss = loss_fn(torch.from_numpy(np.array([G])), val)
        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
