from ..base import Strategy
import numpy as np

class UniformDistribute(Strategy):
    def __repr__(self):
        return "Uniform Distribute"

    def play(self, game):
        assert(game.enable_score==False),"unsupport variant"
        action_dim = game.action_dim
        return np.ones((action_dim,))
