from typing import List
import torch
import numpy as np

from Utils.Utils import Utils


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        if isinstance(obs_dim, List):
            self.obs_buf = []
            self.obs2_buf = []
            for o in obs_dim:
                self.obs_buf.append(
                    np.zeros(Utils.combined_shape(size, o), dtype=np.float32)
                )
                self.obs2_buf.append(
                    np.zeros(Utils.combined_shape(size, o), dtype=np.float32)
                )
        else:
            obs_dim = obs_dim.shape
            self.obs_buf = np.zeros(
                Utils.combined_shape(size, obs_dim), dtype=np.float32
            )
            self.obs2_buf = np.zeros(
                Utils.combined_shape(size, obs_dim), dtype=np.float32
            )
        self.act_buf = np.zeros(Utils.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        if isinstance(obs, List):
            i = 0
            for o in obs:
                self.obs_buf[i][self.ptr] = o
                i += 1
            i = 0
            for o in next_obs:
                self.obs2_buf[i][self.ptr] = o
                i += 1
        else:
            self.obs_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=[o[idxs] for o in self.obs_buf]
            if len(self.obs_buf) != self.max_size
            else self.obs_buf[idxs],
            obs2=[o[idxs] for o in self.obs2_buf]
            if len(self.obs2_buf) != self.max_size
            else self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            if not isinstance(v, List)
            else [torch.as_tensor(o, dtype=torch.float32) for o in v]
            for k, v in batch.items()
        }
