from ..base import Strategy
import numpy as np
class DirectOptimize(Strategy):
    def __init__(self, LAMBDA = 10):
        self.LAMBDA = LAMBDA
        self.EPSILON = 1e-9
    def __repr__(self):
        return "Direct Optimize"

    def reset(self):
        pass

    def play(self, game):
        state = game.get_observation()
        #(t, 4, nr_seller)
        view = state[-1][0]
        trade_amount = state[-1][1]
        trade_value = state[-1][2]
        trade_price = state[-1][3]

        # we use view's value as baseline
        # normalize trade_value and use it to update weight
        weight = self.EPSILON + self.LAMBDA * trade_value
        return weight
