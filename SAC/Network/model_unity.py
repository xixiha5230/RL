from os import access
from unicodedata import name
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    """
    input shape:
            0       [202]
            1       [12,128,128]
    """

    def __init__(self, obs_shapes, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.conv_net1 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
        )

        self.conv_net2 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
        )

        # get fc size only
        with torch.no_grad():
            for o in obs_shapes:
                if len(o) == 3:
                    dummy = torch.zeros(1, *o)
                    break
            x = self.conv_net1(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]

        lin_size = 0
        for o in obs_shapes:
            if len(o) == 1:
                lin_size += o[0]
        # Q1 architecture
        self.lin_net1 = nn.Sequential(
            nn.Linear(lin_size + fc_size + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2 architecture
        self.lin_net2 = nn.Sequential(
            nn.Linear(lin_size + fc_size + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weights_init_)

    def forward(self, state, action):
        for s in state:
            if len(s.shape) > 3:
                conv_x1 = self.conv_net1(s)
                conv_x1 = conv_x1.contiguous()
                conv_x1 = conv_x1.view((conv_x1.shape[0], -1))
                conv_x2 = self.conv_net1(s)
                conv_x2 = conv_x2.contiguous()
                conv_x2 = conv_x2.view((conv_x2.shape[0], -1))
            else:
                lin_x = s
        action = action
        xu1 = torch.cat([conv_x1, lin_x, action], 1)
        xu2 = torch.cat([conv_x2, lin_x, action], 1)
        # xu = torch.cat([state, action], 1)

        x1 = self.lin_net1(xu1)
        x2 = self.lin_net2(xu2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, obs_shapes, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(12, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
        )
        # get fc size only
        with torch.no_grad():
            for o in obs_shapes:
                if len(o) == 3:
                    dummy = torch.zeros(1, *o)
                    break
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]
        # get input linear size
        lin_size = 0
        for o in obs_shapes:
            if len(o) == 1:
                lin_size += o[0]

        self.common_net = nn.Sequential(
            nn.Linear(lin_size + fc_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        for s in state:
            if len(s.shape) > 3:
                conv_x = self.conv_net(s)
                conv_x = conv_x.contiguous()
                conv_x = conv_x.view((conv_x.shape[0], -1))
            else:
                lin_x = s
        x = torch.cat([conv_x, lin_x], 1)
        x = self.common_net(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_shapes, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()

        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(12, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
        )
        # get fc size only
        with torch.no_grad():
            for o in obs_shapes:
                if len(o) == 3:
                    dummy = torch.zeros(1, *o)
                    break
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]
        # get input linear size
        lin_size = 0
        for o in obs_shapes:
            if len(o) == 1:
                lin_size += o[0]

        self.common_net = nn.Sequential(
            nn.Linear(lin_size + fc_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        for s in state:
            if len(s.shape) > 3:
                conv_x = self.conv_net(s)
                conv_x = conv_x.contiguous()
                conv_x = conv_x.view((conv_x.shape[0], -1))
            else:
                lin_x = s
        x = torch.cat([conv_x, lin_x], 1)
        x = self.common_net(x)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


if __name__ == "__main__":
    obs_shapes = [(202,), (12, 128, 128)]
    action_space = 2
    critic = QNetwork(obs_shapes, action_space, 512)
    obs = []
    obs.append(torch.zeros(1, *obs_shapes[0]))
    obs.append(torch.zeros(1, *obs_shapes[1]))
    x1, x2 = critic.forward(obs, [[0, 0]])

    policy = GaussianPolicy(obs_shapes, action_space, 512)
    x3, x4 = policy(obs)
    print()

    dpolicy = DeterministicPolicy(obs_shapes, action_space, 512)
    x5 = dpolicy(obs)
    print()
