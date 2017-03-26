from base import Buyer

class SimpleBuyer(Buyer):
    def __init__(self):
        pass

    def reset(self):
        pass

    def decide_buy_prob(views, prices, qualities):
        #for massive buyers, each view is evenly distributed, no need to sample
        ret = views * (1-prices) * qualities
        return ret
