from typing import List
import numpy as np
import torch
import torch.nn as nn

from Utils.Utils import Utils


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.combine_layer = None
        if isinstance(obs_dim, List):
            for o in obs_dim:
                o = np.array(o)
                if len(o) == 3:
                    self.covnet, self.covnet_q = Utils.mlp_cov2d(
                        o, list(hidden_sizes), activation, act_dim, activation
                    )
                else:
                    self.linq = Utils.mlp(
                        [o[0] + act_dim] + list(hidden_sizes),
                        activation,
                        activation,
                    )
            self.combine_layer = Utils.mlp(
                [list(hidden_sizes)[-1] * 2] + [1],
                activation,
                activation,
            )
        else:
            obs_dim = obs_dim.shape[0]
            self.q = Utils.mlp(
                [obs_dim + act_dim] + list(hidden_sizes) + [1], activation
            )

    def forward(self, obs, act):
        if not self.combine_layer is None:
            o = obs[0] if (obs[0].shape == 4) else obs[1]
            cov_out = self.covnet(o)
            cov_out = torch.cat([cov_out.view((cov_out.shape[0], -1)), act], dim=-1)
            fc_out = self.covnet_q(cov_out)
            o = obs[0] if (obs[0].shape != 4) else obs[1]
            lin_out = self.linq(torch.cat([o, act], dim=-1))
            q = self.combine_layer(torch.cat([fc_out, lin_out], dim=-1))
        else:
            q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.
