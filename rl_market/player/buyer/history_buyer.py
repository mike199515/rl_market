from .base import Buyer
import numpy as np
class HistoryBuyer(Buyer):
    def __init__(self, GAMMA = 0.99):
        self.GAMMA = GAMMA
        self.reset(True)

    def reset(self, hard):
        self.average_amounts=None

    def decide_buy_prob(self, views, prices, trade_amounts):
        if len(trade_amounts)>0:
            latest_trade_amount = trade_amounts[-1]
        else:
            latest_trade_amount = np.ones((views.shape[0],))/1e5
        if self.average_amounts is None:
            self.average_amounts = latest_trade_amount
        else:
            self.average_amounts = self.GAMMA * self.average_amounts + latest_trade_amount
        #log_amounts = np.log(self.average_amounts + 1)
        #norm_average_amounts = log_amounts / np.max(log_amounts)
        norm_average_amounts = .9 + .1 * self.average_amounts / np.max(self.average_amounts)

        return views * (1-prices) * norm_average_amounts
