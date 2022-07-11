import random
import numpy as np
from ReplayMemory import ReplayMemory


class ReplayMemoryIR(ReplayMemory):
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
