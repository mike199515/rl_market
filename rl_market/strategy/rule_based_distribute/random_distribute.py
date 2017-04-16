from ..base import Strategy
import numpy as np

class RandomDistribute(Strategy):
    def __repr(self):
        return "Random Distribute"

    def reset(self):
        pass

    def play(self, game):
        action_dim = game.action_dim
        return np.random.rand(action_dim)
