import numpy as np

class OU(object):
    def function(self, x, mu, theta, sigma):
        return theta*(x-mu) + sigma * np.random.randn(1)
