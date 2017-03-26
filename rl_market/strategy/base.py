
class Strategy(object):
    def __init__(self, game):
        self.game = game

    def train(self, game):
        """[optional] play game simulation to train the model"""
        pass

    def play(self, game):
        """return the desired action for the game"""
        pass
