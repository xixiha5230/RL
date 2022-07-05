from ast import If, arg
import os
from re import L
import numpy
import torch
import torch.nn.functional as F
from torch.optim import Adam
from Algorithm.utils import soft_update, hard_update
from Network.model_unity import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, obs_shapes, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device(
            "cuda" if args.cuda == True and torch.cuda.is_available() else "cpu"
        )

        self.critic = QNetwork(obs_shapes, action_space, args.hidden_size).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(obs_shapes, action_space, args.hidden_size).to(
            self.device
        )
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space).to(self.device)
                ).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(
                obs_shapes, action_space, args.hidden_size, None
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                obs_shapes, action_space, args.hidden_size, None
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        nst = []
        for s in state:
            t = torch.FloatTensor(numpy.array(s)).to(self.device)
            if len(t.shape) > 3:
                t = t.transpose(2, 3).transpose(1, 2)
            nst.append(t)
        state = nst
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)

        action = action.detach().cpu().numpy()
        return action

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch,
        ) = memory.sample(batch_size=batch_size)

        for i in range(len(state_batch)):
            state_batch[i] = (
                torch.FloatTensor(numpy.array(state_batch[i]))
                .squeeze(1)
                .to(self.device)
            )
            if len(state_batch[i].shape) > 3:
                state_batch[i] = state_batch[i].transpose(2, 3).transpose(1, 2)
            next_state_batch[i] = (
                torch.FloatTensor(numpy.array(next_state_batch[i]))
                .squeeze(1)
                .to(self.device)
            )
            if len(next_state_batch[i].shape) > 3:
                next_state_batch[i] = (
                    next_state_batch[i].transpose(2, 3).transpose(1, 2)
                )
        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = (
            torch.FloatTensor(numpy.array(action_batch)).squeeze(1).to(self.device)
        )
        reward_batch = (
            torch.FloatTensor(numpy.array(reward_batch)).to(self.device).unsqueeze(1)
        )
        mask_batch = (
            torch.FloatTensor(numpy.array(mask_batch)).to(self.device).unsqueeze(1)
        )
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if ckpt_path is None:
            path = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists(path + "/checkpoints/" + env_name):
                os.makedirs(path + "/checkpoints/" + env_name)
            ckpt_path = path + "/checkpoints/{}/sac_checkpoint_{}".format(
                env_name, suffix
            )
        else:
            ckpt_path += "/checkpoints/" + env_name
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            ckpt_path += "/sac_checkpoint_{}".format(suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()