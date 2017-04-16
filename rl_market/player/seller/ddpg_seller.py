from .base import Seller

class DDPGSeller(Seller):
    def __init__(self, ddpg_strategy):
        self.ddpg_strategy = ddpg_strategy

    def decide_price(self, game):
        action = self.ddpg_strategy.play(game)
        return action

    def train(self, game):
        #code
        pass
