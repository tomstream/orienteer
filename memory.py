import time
import numpy as np
import random

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def append(self, data):
        if len(self.memory) < self.capacity:
            self.memory.append(data)
        else:
            self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class TimeCount(object):
    def __init__(self):
        self.data_ = []

    def start(self):
        self.data_.append(time.time())

    def end(self):
        self.data_[-1] = time.time() - self.data_[-1]

    def mean(self):
        return np.mean(self.data_)
