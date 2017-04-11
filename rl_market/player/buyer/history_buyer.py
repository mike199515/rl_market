from .base import Buyer
import numpy as np
class HistoryBuyer(Buyer):
    def __init__(self, GAMMA = 0.95):
        self.GAMMA = GAMMA
        self.reset(True)

    def reset(self, hard):
        self.average_views=None

    def decide_buy_prob(self, views, prices, qualities):
        if self.average_views is None:
            self.average_views = views
        else:
            self.average_views = self.GAMMA * self.average_views + views
            self.average_views = self.average_views / np.sum(self.average_views)
        return self.average_views * (1-prices)
