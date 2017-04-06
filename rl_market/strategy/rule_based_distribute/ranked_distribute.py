from ..base import Strategy
import numpy as np

"""
try rank each seller and distribute fix amount view
"""
class RankedDistribute(Strategy):
    def __init__(self):
        pass
    
    def play(self, game):
        state = game.get_observation()
        #(t, 4, nr_seller)
        view = state[-1][0]
        trade_amount = state[-1][1]
        trade_value = state[-1][2]
        trade_price = state[-1][3]
        
        array = np.array(view)
        order = array.argsort()
        ranks = order.argsort()
        
        weight = 1./np.exp(ranks)
        return weight
        
