import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from dataclasses import dataclass
from typing import Any
from collections import deque
import random
import wandb
from tqdm import tqdm


@dataclass
class Sarsd:
    obs: Any
    action: int
    reward: float
    next_obs: Any
    done: bool


class ReplayBuffer():

    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.queue = deque(maxlen=self.buffer_size)

    def inser(self, sarsd):
        self.queue.append(sarsd)

    def sample(self, sample_num):
        return random.sample(self.queue, sample_num)


class ConvModel(nn.Module):

    def __init__(self, obs_shape, action_num, learning_rate):
        assert len(obs_shape) == 3  #chanel,height and width
        super(ConvModel, self).__init__()
        self.obs_shape = obs_shape
        self.action_num = action_num
        self.learning_rate = learning_rate
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, (8, 8), stride=(4, 4)), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, (4, 4), stride=(2, 2)), torch.nn.ReLU())
        with torch.no_grad():
            dummy = torch.zeros(1, *self.obs_shape)
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]
        self.fc_net = torch.nn.Sequential(
            torch.nn.Linear(fc_size, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, self.action_num))
        self.opt = torch.optim.Adam(self.fc_net.parameters(), lr=learning_rate)

    def forward(self, x):
        conv_x = self.conv_net(x / 255.0)
        conv_x = conv_x.view((conv_x.shape[0], -1))
        return self.fc_net(conv_x)


class Agent():

    def __init__(self, model: ConvModel, target_model: ConvModel, device):
        self.model = model
        self.target_model = target_model
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.device = device

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self, batch_data):
        obs_batch = torch.stack([torch.Tensor(s.obs)
                                 for s in batch_data]).to(self.device)
        reward_batch = torch.stack(
            [torch.Tensor([s.reward]) for s in batch_data]).to(self.device)
        done_batch = torch.stack([
            torch.Tensor([0]) if s.done else torch.Tensor([1])
            for s in batch_data
        ]).to(self.device)
        next_obs_batch = torch.stack(
            [torch.Tensor(s.next_obs) for s in batch_data]).to(self.device)
        act_batch = [s.action for s in batch_data]

        with torch.no_grad():
            next_action_q = self.target_model(next_obs_batch).max(-1)[0]

        self.model.opt.zero_grad()
        action_q = self.model(obs_batch)
        one_hot_action = F.one_hot(torch.LongTensor(act_batch),
                                   self.model.action_num).to(self.device)
        # torch sum 是因为把one_hot乘出来后的0去除，x+0+0
        # loss = ((reward_batch + done_batch[:, 0] * next_action_q -
        #          torch.sum(action_q * one_hot_action, -1))**2).mean()
        loss = self.loss_fn(
            torch.sum(action_q * one_hot_action, -1),
            reward_batch[:, 0] + done_batch[:, 0] * next_action_q)
        loss.backward()
        self.model.opt.step()
        return loss


def train(agent: Agent, env):
    obs = env.reset()
    #环境随机采集数量
    min_env_step = 10000
    sample_zie = 2500

    step = -min_env_step
    train_step = 1
    train_after_step = 500
    update_after_tran = 50

    total_reward = 0
    reward_batch = []

    agent.update_target_model()

    export_rate = 0.9999
    export_rate_min = 0.01
    esp = 1

    tq = tqdm()
    try:
        while True:
            step += 1
            if train_step > 0 and esp > export_rate_min:
                esp = export_rate**train_step
            if random.uniform(0, 1) < esp:
                action = env.action_space.sample()
            else:
                action = agent.model(
                    torch.Tensor(obs).unsqueeze(0).to(
                        agent.device)).max(-1)[-1].item()

            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            reward = reward / 100.0
            rb.inser(Sarsd(obs, action, reward, next_obs, done))

            obs = next_obs

            if done:
                obs = env.reset()
                reward_batch.append(total_reward)
                total_reward = 0

            if len(rb.queue) > min_env_step and step % train_after_step == 0:
                loss = agent.learn(rb.sample(sample_zie))
                tq.update(1)
                wandb.log(
                    {
                        "export": esp,
                        "loss": loss.detach().item(),
                        "reward": numpy.mean(reward_batch)
                    },
                    step=train_step)
                reward_batch = []
                train_step += 1
                if train_step % update_after_tran == 0:
                    agent.update_target_model()
                    torch.save(agent.target_model,
                               f'./DQN/model/{train_step}.pth')
                    print("update model", train_step)

    except KeyboardInterrupt:
        pass


from utils import FrameStackingAndResizingEnv
if __name__ == "__main__":
    repla_buffer_size = 100000
    wandb.init(project="dqn", name='break-out')
    env = gym.make("Breakout-v0")
    env = FrameStackingAndResizingEnv(env, 84, 84)
    rb = ReplayBuffer(repla_buffer_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    torch.cuda.empty_cache()
    model = ConvModel(env.observation_space.shape, env.action_space.n, 0.01)
    model.to(device)
    target_model = ConvModel(env.observation_space.shape, env.action_space.n,
                             0.01)
    target_model.to(device)
    agent = Agent(model, target_model, device)
    train(agent, env)
