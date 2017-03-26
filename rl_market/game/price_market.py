from base import Game
import numpy as np

class PriceMarket(Game):

    def __init__(self,
            sellers,
            buyer,
            max_duration):
        super(PriceMarket,self).__init__()

        self.max_duration = max_duration
        self.sellers = sellers
        self.buyer = buyer
        self.reset()
        pass

    def reset(self):
        self.duration = 0
        self.history_length = 0
        self.view=[]
        self.trade_amount=[]
        self.trade_value=[]
        self.price=[]

        for seller in self.sellers:
            seller.reset()
        self.buyer.reset()

    def get_observation(self):
        if self.history_length == 0:
            pass

    def step(self, weights):
        assert(weights.shape[0] == len(self.ellers)),"weight mismatch"
        #get seller's price & quality
        nr_sellers = len(self.sellers)

        price = np.zeros((nr_sellers,))
        for i, seller in enumerate(self.sellers):
            price[i] = seller.decide_price(self, i)

        quality = np.zeros((nr_sellers,))
        for i, seller in enumerate(self.sellers):
            quality[i] = seller.get_quality(self, i)

        weights = weights/np.sum(weights) # normalize
        # calculate views
        view = weights[:]
        # calculate trade

        trade_amount = self.buyer.decide_buy_prob(views=view, prices=price, qualities=quality)
        trade_value = trade_amount * price

        #write to history
        self.view+=view
        self.trade_amount+=trade_amount
        self.trade_value+=trade_value
        self.price+=price
        self.duration += 1
        reward = self._calculate_reward()
        done = (self.duration > self.max_duration)
        return reward, done

    def _calculate_reward(self):
        trade_value = self.trade_value[-1]
        return np.sum(trade_value)
