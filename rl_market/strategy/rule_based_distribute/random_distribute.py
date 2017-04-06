from ..base import Strategy
import numpy as np

class RandomDistribute(Strategy):
    def play(self, game):
        action_dim = game.action_dim
        return np.random.rand(action_dim)
