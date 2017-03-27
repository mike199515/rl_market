from .base import Seller
import numpy as np

class SimpleSeller(Seller):
    def __init__(self, quality_sampler, cost_sampler, price_sampler,
            noise_sampler,
            discount_factor = 0.95
            ):
        self.quality_sampler=quality_sampler
        self.cost_sampler=cost_sampler
        self.price_sampler=price_sampler
        self.noise_sampler=noise_sampler

        self.discount_factor = discount_factor

        self.reset(hard = True)

    def reset(self, hard):
        if hard:
            self.quality = self.quality_sampler.sample()
            self.cost = self.cost_sampler.sample()
        self.trade_history = []

    def decide_price(self, game, index):
        nr_history = game.duration
        if nr_history <=2:
            return self.price_sampler.sample()
        last_price = game.price[-1][index]
        last_trade_amount = game.trade_amount[-1][index]
        last_profit = (last_price - self.cost) * last_trade_amount
        self.trade_history.append((last_price, last_profit))
        if len(self.trade_history)> 20:
            self.trade_history=self.trade_history[1:]
        #find the price most successful and sample from that range
        best_price, best_profit = None, 0
        for idx, (price, profit) in enumerate(self.trade_history):
            interval = len(self.trade_history) - idx
            discounted_profit = profit * np.power(self.discount_factor, interval)
            if discounted_profit > best_profit:
                best_price = price
                best_profit = profit
        if best_price is None:
            return self.price_sampler.sample()
        return self._regularize(best_price + self.noise_sampler.sample())

    def get_quality(self, game, index):
        return self.quality

    def _regularize(self, val):
        return min(max(0.,val),1.)
