import torch
import torch.nn as nn
import numpy as np
from urllib3 import Retry


class Utils:
    def combined_shape(length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def mlp(sizes, activation, output_activation=nn.Identity):
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        return nn.Sequential(*layers)

    def mlp_cov2d(
        obs_size, fc_size, activation, act_size=0, output_activation=nn.Identity
    ):
        layers = []
        conv_net = nn.Sequential(
            nn.Conv2d(obs_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_size)
            x = conv_net(dummy)
            s = x.shape
            size = s[1] * s[2] * s[3]
        fc_size[0] = size + act_size
        for j in range(len(fc_size) - 1):
            act = activation if j < len(fc_size) - 2 else output_activation
            layers += [nn.Linear(fc_size[j], fc_size[j + 1]), act()]
        return conv_net, nn.Sequential(*layers)

    def count_vars(module):
        return sum([np.prod(p.shape) for p in module.parameters()])

    def transpose(obs):
        for i in range(len(obs)):
            if len(obs[i].shape) == 4:
                obs[i] = (
                    torch.tensor(obs[i])
                    .transpose(2, 3)
                    .transpose(1, 2)
                    .numpy()
                    .squeeze()
                )
                # obs[i] = obs[i].transpose(2, 3).transpose(1, 2)
        return obs
