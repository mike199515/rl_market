from ..base import Strategy
import numpy as np

class UCB1(Strategy):
    def __init__(self, nr_seller):
        self.nr_seller = nr_seller
        self.reset()
    def __repr__(self):
        return "UCB1"
    def reset(self):
        self.game_round = 0
        self.last_selection = None
        self.value = np.zeros((self.nr_seller,))
        self.count = np.zeros((self.nr_seller,))

    def _one_hot(self,idx):
        ret = np.zeros((self.nr_seller,))
        ret[idx] = 1
        return ret

    def play(self, game):
        #update last trade round
        if self.last_selection is not None:
            self.count[self.last_selection]+=1.
            state = game.get_observation()
            view = state[-1][0]
            trade_value = state[-1][2]
            reward = trade_value[self.last_selection]

            n = self.count[self.last_selection]
            self.value[self.last_selection]= (n-1.)/n*self.value[self.last_selection]+1./n*reward

        if self.game_round < self.nr_seller:
            self.last_selection = self.game_round
        else:
            w_values = np.array(self.value)
            for i in range(game.action_dim):
                w_values[i] += np.sqrt(2*np.log(self.game_round))/self.count[i]
            self.last_selection = np.argmax(w_values)
        self.game_round+=1
        return self._one_hot(self.last_selection)
