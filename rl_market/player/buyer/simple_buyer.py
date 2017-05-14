from .base import Buyer

class SimpleBuyer(Buyer):
    def __init__(self):
        pass

    def reset(self, hard):
        pass

    def decide_buy_prob(self, views, prices, trade_amounts):
        #for massive buyers, each view is evenly distributed, no need to sample
        ret = views * (1-prices)
        return ret
