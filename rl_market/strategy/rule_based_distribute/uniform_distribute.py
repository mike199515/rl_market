from ..base import Strategy
import numpy as np

class UniformDistribute(Strategy):
    def play(self, game):
        action_dim = game.action_dim
        return np.ones((action_dim,))
