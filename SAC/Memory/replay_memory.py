import random
import numpy as np
import pandas as pd
import os
import pickle

# TODO Save buffer


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))

        st = [i for item in state for i in item]
        nst = [i for item in next_state for i in item]
        state_num = len(state[0])
        state = []
        next_state = []
        for i in range(state_num):
            state.append(st[i::state_num])
            next_state.append(nst[i::state_num])

        for s in state:
            s = np.stack(s)
        for s in next_state:
            s = np.stack(s)
        action = np.stack(action)
        reward = np.stack(reward)
        done = np.stack(done)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print("Saving buffer to {}".format(save_path))

        with open(save_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print("Loading buffer from {}".format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
