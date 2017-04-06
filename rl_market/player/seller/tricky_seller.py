from .base import Seller
import numpy as np

class TrickySeller(SimpleSeller):
    def __init__(self, trick_prob = 0.3, *args, **kargs):
        self.trick_prob = trick_prob
        super(self, ComparativeSeller).__init__(*args, **kargs)
    
    def decide_price(self, game, index):
        nr_history = game.duration
        
        if nr_history <=2:
            return self.price_sampler.sample()
        
        # trick the direct_algorithm by randomly pick 0 price to attract buyers
        if np.random.random()<self.trick_prob:
            return 0.
        
        # pick up the most profitable index instead
        best_price = None
        best_profit = -float("inf")
        for price, trade_amount in zip(game.price[-1], game.trade_amount[-1]):
            profit = (last_price - self.cost) * last_trade_amount
            if profit > best_profit:
                best_price = price
                best_profit = profit
        self.trade_history.append((best_price, best_profit))
        
        if len(self.trade_history)> self.max_trade_history:
            self.trade_history=self.trade_history[1:]
        #find the price most successful and sample from that range
        best_price, best_profit = None, 0
        for t, (price, profit) in enumerate(self.trade_history):
            interval = len(self.trade_history) - t
            discounted_profit = profit * np.power(self.discount_factor, interval)
            if discounted_profit > best_profit:
                best_price = price
                best_profit = profit
        if best_price is None:
            return self.price_sampler.sample()
        return self._regularize(best_price + self.noise_sampler.sample())
