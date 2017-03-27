from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, buff_size):
        self.buff_size = buff_size
        self.nr_experience = 0
        self.buff = deque()

    def get_batch(self, batch_size):
        return random.sample(self.buff, min(self.nr_experience,batch_size))

    def add(self, experience):
        if self.nr_experience < self.buff_size:
            self.nr_experience+=1
        else:
            self.buff.popleft()
        self.buff.append(experience)

    def reset(self):
        self.buff=deque()
        self.nr_experience=0
