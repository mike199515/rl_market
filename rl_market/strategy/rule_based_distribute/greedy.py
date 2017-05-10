from ..base import Strategy
import numpy as np
class Greedy(Strategy):
    def __init__(self):
        pass
    def __repr__(self):
        return "Greedy"

    def reset(self):
        pass

    def _one_hot(self, idx, length):
        ret = np.zeros((length,))
        ret[idx] = 1
        return ret

    def play(self, game):
        state = game.get_observation()
        #(t, 4, nr_seller)
        view = state[-1][0]
        trade_amount = state[-1][1]
        trade_value = state[-1][2]
        trade_price = state[-1][3]

        # normalize trade_value and use it to update weight
        return self._one_hot(np.argmax(trade_value),len(trade_value))
