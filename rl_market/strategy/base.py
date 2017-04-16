
class Strategy(object):

    def resest(self):
        """[optional] reset the stateful strategy"""
        pass

    def train(self, game):
        """[optional] play game simulation to train the model"""
        pass

    def play(self, game):
        """return the desired action for the game"""
        pass
