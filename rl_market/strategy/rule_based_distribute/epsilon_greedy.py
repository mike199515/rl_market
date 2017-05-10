from ..base import Strategy
import numpy as np

class EpsilonGreedy(Strategy):
    def __init__(self, action_dim, epsilon = 0.1):
        self.epsilon=epsilon
        self.action_dim = action_dim
        self.reset()

    def __repr(self):
        return "Epsilon Greedy"

    def reset(self):
        self.game_round = 0
        self.value = np.zeros((self.action_dim,))
        self.count = np.zeros((self.action_dim,))
        self.last_action = None

    def _one_hot(self, idx):
        ret = np.zeros((self.action_dim,))
        ret[idx] = 1
        return ret

    def play(self, game):
        if self.last_action is not None:
           state = game.get_observation()
           reward = state[-1][2][self.last_action]
           self.count[self.last_action]+=1.
           n = self.count[self.last_action]
           self.value[self.last_action] = self.value[self.last_action] * (n-1)/n + reward/n

        if self.epsilon < np.random.rand():
            self.last_action = np.random.randint(self.action_dim-1)
        else:
            self.last_action = np.argmax(self.value)
        return self._one_hot(self.last_action)
