import numpy as np

class Sampler(object):
    def sample(self):
        pass

class UniformSampler(Sampler):
    def __init__(self, low = 0., high=1.):
        self.low = low
        self.dur = high - low

    def sample(self):
        return self.low + self.dur * np.random.rand()

class GaussianSampler(Sampler):
    def __init__(self, mu, sigma):
        self.sigma = sigma
        self.mu = mu

    def sample(self):
        return self.mu + self.sigma * np.random.randn()

class ClipGaussianSampler(Sampler):
    def __init__(self, mu, sigma, low= 0., high = 1.):
        self.sigma = sigma
        self.mu = mu
        self.low = low
        self.high = high

    def sample(self):
        val = self.mu + self.sigma * np.random.randn()
        return min(max(self.low, val), self.high)

class BoundGaussianSampler(Sampler):
    def __init__(self, mu, sigma,  low= 0., high = 1.):
        self.sigma = sigma
        self.mu = mu
        self.low = low
        self.high = high

    def sample(self):
        while(True):
            val = self.mu + self.sigma * np.random.randn()
            if self.low <=val<=self.high:
                return val
