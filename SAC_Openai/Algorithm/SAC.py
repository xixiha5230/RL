from copy import deepcopy
import numpy as np
import itertools
import time
import torch
from torch.optim import Adam

from ReplayBuffer.ReplayBuffer import ReplayBuffer
from Module.MLPActorCritic import MLPActorCritic
from Utils.Utils import Utils


class SAC:
    def __init__(
        self,
        env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=4000,
        epochs=100,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        alpha=0.2,
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        num_test_episodes=10,
        max_ep_len=1000,
        logger_kwargs=dict(),
        save_freq=1,
    ):
        self.env_fn = env_fn
        self.actor_critic = actor_critic
        self.ac_kwargs = ac_kwargs
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.logger_kwargs = logger_kwargs
        self.save_freq = save_freq

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data, ac, ac_targ):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, ac):
        o = data["obs"]
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update(self, data, q_optimizer, q_params, pi_optimizer, ac, ac_targ):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data, ac, ac_targ)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data, ac)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        # logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, ac, o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent(self, ac, test_env):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(self.get_action(ac, o, True))
                ep_ret += r
                ep_len += 1
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            print("test EpRet:", ep_ret, " EpLen", ep_len)

    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        env, test_env = self.env_fn(), self.env_fn()
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = env.action_space.high[0]

        # Create actor-critic module and target networks
        ac = self.actor_critic(
            env.observation_space, env.action_space, **self.ac_kwargs
        )
        ac_targ = deepcopy(ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

        # Experience buffer
        replay_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=self.replay_size
        )

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(Utils.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
        # logger.log("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)
        print("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

        pi_optimizer = Adam(ac.pi.parameters(), lr=self.lr)
        q_optimizer = Adam(q_params, lr=self.lr)

        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, ep_ret, ep_len = env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > self.start_steps:
                a = self.get_action(ac, o)
            else:
                a = env.action_space.sample()

            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                # logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = replay_buffer.sample_batch(self.batch_size)
                    self.update(batch, q_optimizer, q_params, pi_optimizer, ac, ac_targ)

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    pass
                # logger.save_state({"env": env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent(ac, test_env)

                # Log info about epoch
                # logger.log_tabular("Epoch", epoch)
                # logger.log_tabular("EpRet", with_min_and_max=True)
                # logger.log_tabular("TestEpRet", with_min_and_max=True)
                # logger.log_tabular("EpLen", average_only=True)
                # logger.log_tabular("TestEpLen", average_only=True)
                # logger.log_tabular("TotalEnvInteracts", t)
                # logger.log_tabular("Q1Vals", with_min_and_max=True)
                # logger.log_tabular("Q2Vals", with_min_and_max=True)
                # logger.log_tabular("LogPi", with_min_and_max=True)
                # logger.log_tabular("LossPi", average_only=True)
                # logger.log_tabular("LossQ", average_only=True)
                # logger.log_tabular("Time", time.time() - start_time)
                # logger.dump_tabular()
                print("Epoch:", epoch, " EpRet:", ep_ret)
