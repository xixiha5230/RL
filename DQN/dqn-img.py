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


class Model(nn.Module):
    def __init__(self, obs_num, action_num, learning_rate):
        super(Model, self).__init__()
        self.obs_num = obs_num
        self.action_num = action_num
        self.learning_rate = learning_rate
        self.net = nn.Sequential(nn.Linear(self.obs_num, 512), nn.ReLU(),
                                 nn.Linear(512, self.action_num))
        self.opt = torch.optim.Adam(params=self.net.parameters(),
                                    lr=self.learning_rate)

    # 有这个forward就能直接：model(torch.Tensor(obs))
    # 不用：model.net(torch.Tensor(obs))
    # 原因不知道
    def forward(self, x):
        return self.net(x)


class Agent():
    def __init__(self, model: Model, target_model: Model):
        self.model = model
        self.target_model = target_model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self, batch_data):
        obs_batch = torch.stack([torch.Tensor(s.obs) for s in batch_data])
        reward_batch = torch.stack(
            [torch.Tensor([s.reward]) for s in batch_data])
        done_batch = torch.stack([
            torch.Tensor([0]) if s.done else torch.Tensor([1])
            for s in batch_data
        ])
        next_obs_batch = torch.stack(
            [torch.Tensor(s.next_obs) for s in batch_data])
        act_batch = [s.action for s in batch_data]

        with torch.no_grad():
            next_action_q = self.target_model(next_obs_batch).max(-1)[0]

        self.model.opt.zero_grad()
        action_q = self.model(obs_batch)
        one_hot_action = F.one_hot(torch.LongTensor(act_batch),
                                   self.model.action_num)
        # torch sum 是因为把one_hot乘出来后的0去除，x+0+0
        loss = ((reward_batch + done_batch[:, 0] * next_action_q -
                 torch.sum(action_q * one_hot_action, -1))**2).mean()
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
    update_after_tran = 20

    total_reward = 0
    reward_batch = []

    agent.update_target_model()

    export_rate = 0.99985
    export_rate_min = 0.01
    esp = 1

    try:
        while True:
            step += 1
            if train_step > 0 and esp > export_rate_min:
                esp = export_rate**train_step
            if random.uniform(0, 1) < esp:
                action = env.action_space.sample()
            else:
                action = agent.model(torch.Tensor(obs)).max(-1)[-1].item()

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


if __name__ == "__main__":
    repla_buffer_size = 100000
    wandb.init(project="dqn", name='cartpol')
    env = gym.make("CartPole-v1")
    rb = ReplayBuffer(repla_buffer_size)
    model = Model(env.observation_space.shape[0], env.action_space.n, 0.01)
    target_model = Model(env.observation_space.shape[0], env.action_space.n,
                         0.01)
    agent = Agent(model, target_model)
    train(agent, env)
