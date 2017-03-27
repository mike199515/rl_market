from ..base import Strategy
import numpy as np
class DirectOptimize(Strategy):
    def __init__(self, LAMBDA = 10):
        self.LAMBDA = LAMBDA

    def play(self, game):
        state = game.get_observation()
        #(4, nr_seller)
        view = state[0]
        trade_amount = state[1]
        trade_value = state[2]
        trade_price = state[3]

        # we use view's value as baseline
        # normalize trade_value and use it to update weight
        weight = view + self.LAMBDA * trade_value / np.sum(trade_value)
        return weight
