from ..base import Strategy
import numpy as np
class AccumOptimize(Strategy):
    def __init__(self, LAMBDA = 10, GAMMA=0.99):
        self.LAMBDA = LAMBDA
        self.GAMMA = GAMMA
        self.reset()

    def __repr__(self):
        return "Accum Optimize"

    def reset(self):
        self.accum_trade_value = None

    def play(self, game):
        state = game.get_observation()
        #(t, 4 ,nr_seller)
        trade_value = state[-1][2]
        if self.accum_trade_value is not None:
            self.accum_trade_value=trade_value+self.GAMMA * self.accum_trade_value
        else:
            self.accum_trade_value=trade_value

        weight = self.accum_trade_value / np.sum(self.accum_trade_value)
        return weight
