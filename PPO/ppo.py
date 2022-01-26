from dis import dis
import imp

import os
from select import select
from textwrap import indent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        # real action we took
        self.actions = []
        # 预测概率
        self.probs = []
        # critic values
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        num_states = len(self.states)
        # np.arange生成数组
        batch_start = np.arange(0, num_states, self.batch_size)
        indices = np.arange(num_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs), \
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class ActorNetwork(nn.Module):
    def __init__(self,
                 num_actions,
                 input_dims,
                 lr,
                 fc1_dims=256,
                 fc2_dims=256,
                 chkpt_dir='PPO/model'):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, 'actor')
        self.actor = nn.Sequential(nn.Linear(*input_dims, fc1_dims), nn.ReLU(),
                                   nn.Linear(fc1_dims, fc2_dims), nn.ReLU(),
                                   nn.Linear(fc2_dims, num_actions),
                                   nn.Softmax(dim=-1))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # self.device = torch.device(
        #     'cuda:0' if torch.cuda.is_available else 'cpu')
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_chkpt(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_chkpt(self, chkpt_file):
        self.load_state_dict(torch.load(chkpt_file))


class CriticNetwork(nn.Module):
    def __init__(self,
                 input_dims,
                 lr,
                 fc1_dims=256,
                 fc2_dims=256,
                 chkpt_dir='PPO/model'):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, 'critic')
        self.critic = nn.Sequential(nn.Linear(*input_dims, fc1_dims),
                                    nn.ReLU(), nn.Linear(fc1_dims, fc2_dims),
                                    nn.ReLU(), nn.Linear(fc2_dims, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # self.device = torch.device(
        #     'cuda:0' if torch.cuda.is_available else 'cpu')
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state):
        val = self.critic(state)
        return val

    def save_chkpt(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_chkpt(self, chkpt_file):
        self.load_state_dict(torch.load(chkpt_file))


class Agent:
    def __init__(self,
                 num_action,
                 input_dims,
                 gamma=0.99,
                 lr=0.0003,
                 gae_lambda=0.95,
                 policy_clip=0.1,
                 batch_size=64,
                 N=2048,
                 num_epochs=10):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.num_epochs = num_epochs

        self.actor = ActorNetwork(num_action, input_dims, lr)
        self.critic = CriticNetwork(input_dims, lr)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_chkpt()
        self.critic.save_chkpt()

    def load_models(self, chkpt_file_actor, chkpt_file_critic):
        print("... loading models ...")
        self.actor.load_chkpt(chkpt_file_actor)
        self.critic.load_chkpt(chkpt_file_critic)

    def choose_action(self, obs):
        obs = torch.tensor([obs], dtype=torch.float).to(self.actor.device)

        dist = self.actor(obs)
        value = self.critic(obs)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.num_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches(
            )

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # 计算advantage:At，视频在 -> 42:30
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] +
                                       self.gamma * values[k + 1] *
                                       (1 - int(done_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch],
                                      dtype=torch.float).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(
                    self.actor.device)

                dist = self.actor(states)
                critic_val = self.critic(states)

                critic_val = torch.squeeze(critic_val)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs / old_probs).exp()
                weighted_prob = advantage[batch] * prob_ratio
                weighted_clipped_prob = torch.clamp(
                    prob_ratio, 1 - self.policy_clip,
                    1 + self.policy_clip) * advantage[batch]

                actor_loss = -torch.min(weighted_prob,
                                        weighted_clipped_prob).mean()

                returns = advantage[batch] + values[batch]

                critic_loss = (returns - critic_val)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
