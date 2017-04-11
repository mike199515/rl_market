from .base import Game
import numpy as np

class PriceMarket(Game):

    def __init__(self,
            sellers,
            buyer,
            max_duration,
            nr_observation = 1):
        super(PriceMarket,self).__init__()

        self.max_duration = max_duration
        self.sellers = sellers
        self.buyer = buyer
        self.nr_observation = nr_observation
        self.reset()
        pass

    def reset(self, hard = False):
        self.state_shape = (self.nr_observation, 4, len(self.sellers))
        self.action_dim = len(self.sellers)
        self.duration = 0
        self.view=[]
        self.trade_amount=[]
        self.trade_value=[]
        self.price=[]

        for seller in self.sellers:
            seller.reset(hard)
        self.buyer.reset(hard)
        #we step for nr_observation round for observation to be valid
        for i in range(self.nr_observation):
            self.step(np.ones(len(self.sellers)))

    def get_observation(self):
        if len(self.view) < self.nr_observation:
            assert(False),"no observation available"
        states = []
        for i in range(self.nr_observation):
            last_view=self.view[-1-i]
            last_trade_amount=self.trade_amount[-1-i]
            last_trade_value=self.trade_value[-1-i]
            last_price=self.price[-1-i]
            #print(last_view, last_trade_amount, last_trade_value, last_price)
            state = np.array((last_view,last_trade_amount,last_trade_value,last_price))
            states.append(state)

        states = np.array(states)

        return states

    def get_observation_string(self):
        assert(False),"obsoleted"
        state  = self.get_observation()
        ret = " view {}:{}\n trade_amount {}:{}\n trade_value {}:{}\n price {}:{}".\
                 format(np.mean(state[0]), np.std(state[0]),
                        np.mean(state[1]), np.std(state[1]),
                        np.mean(state[2]), np.std(state[2]),
                        np.mean(state[3]), np.std(state[3]))
        return ret

    def step(self, weights):
        assert(len(weights.shape)==1)
        assert(weights.shape[0] == len(self.sellers)),"weight mismatch"
        #print("weight:",weights)
        #get seller's price & quality
        nr_sellers = len(self.sellers)

        price = np.zeros((nr_sellers,))
        for i, seller in enumerate(self.sellers):
            price[i] = seller.decide_price(self, i)

        quality = np.zeros((nr_sellers,))
        for i, seller in enumerate(self.sellers):
            quality[i] = seller.get_quality(self, i)

        weights = np.maximum(weights,0)
        weights = weights/np.sum(weights) # normalize
        # calculate views
        view = weights[:]
        # calculate trade
        trade_amount = self.buyer.decide_buy_prob(views=view, prices=price, qualities=quality)
        trade_value = trade_amount * price

        #write to history
        self.view.append(view)
        self.trade_amount.append(trade_amount)
        self.trade_value.append(trade_value)
        self.price.append(price)
        self.duration += 1
        reward = self._calculate_reward()
        done = (self.duration > self.max_duration)
        return reward, done

    def _calculate_reward(self):
        trade_value = self.trade_value[-1]
        return np.sum(trade_value)
