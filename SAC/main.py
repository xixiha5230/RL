import argparse
from fileinput import filename
import os
import glob
import datetime
from tkinter.messagebox import NO
import gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from Memory.replay_memory import ReplayMemory
from Algorithm.sac import SAC
from Envwrapper.UnityEnv import UnityWrapper

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
parser.add_argument(
    "--env-name",
    default="605",
    help="Mujoco Gym environment (default: LunarLander-v2)",
)
parser.add_argument(
    "--policy",
    default="Gaussian",
    help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
)
parser.add_argument(
    "--eval",
    type=bool,
    default=True,
    help="Evaluates a policy a policy every 10 episode (default: True)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor for reward (default: 0.99)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    metavar="G",
    help="target smoothing coefficient(τ) (default: 0.005)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0003,
    metavar="G",
    help="learning rate (default: 0.0003)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.2,
    metavar="G",
    help="Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)",
)
parser.add_argument(
    "--automatic_entropy_tuning",
    type=bool,
    default=False,
    metavar="G",
    help="Automaically adjust α (default: False)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123456,
    metavar="N",
    help="random seed (default: 123456)",
)
parser.add_argument(
    "--batch_size", type=int, default=512, metavar="N", help="batch size (default: 256)"
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1000001,
    metavar="N",
    help="maximum number of steps (default: 1000000)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    metavar="N",
    help="hidden size (default: 256)",
)
parser.add_argument(
    "--updates_per_step",
    type=int,
    default=1,
    metavar="N",
    help="model updates per simulator step (default: 1)",
)
parser.add_argument(
    "--start_steps",
    type=int,
    default=10000,
    metavar="N",
    help="Steps sampling random actions (default: 10000)",
)
parser.add_argument(
    "--target_update_interval",
    type=int,
    default=1,
    metavar="N",
    help="Value target update per no. of updates per step (default: 1)",
)
parser.add_argument(
    "--replay_size",
    type=int,
    default=1000000,
    metavar="N",
    help="size of replay buffer (default: 10000000)",
)
parser.add_argument(
    "--resume", type=bool, default=False, help="Resume training (default: False)"
)
parser.add_argument(
    "--cuda", type=bool, default=True, help="Resume on GPU (default: False)"
)
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Environment
# env = gym.make(args.env_name, continuous=(args.policy == "Gaussian"))
file_name = "venv_605/"
env = UnityWrapper(
    train_mode=True, file_name=file_name, no_graphics=True, seed=args.seed
)
obs_shapes, discrete_action_size, continuous_action_size = env.get_size()
obs_shapes[1] = tuple(reversed(obs_shapes[1]))
# Agent
if args.policy == "Gaussian":
    agent = SAC(obs_shapes, continuous_action_size, args)
else:
    agent = SAC(obs_shapes, discrete_action_size, args)

if args.resume is True:
    list_of_files = glob.glob(
        os.path.dirname(os.path.abspath(__file__))
        + "/checkpoints/"
        + args.env_name
        + "/*"
    )
    latest_file = max(list_of_files, key=os.path.getctime)
    print("Resume training load: ", latest_file)
    agent.load_checkpoint(latest_file)

# Tesnorboard
if args.resume is True:
    files = glob.glob(
        os.path.dirname(os.path.abspath(__file__)) + "/runs/" + args.env_name + "/*"
    )
    path = max(files, key=os.path.getctime)
    writer = SummaryWriter(path)
    event_acc = EventAccumulator(path, size_guidance={"scalars": 0})
    event_acc.Reload()
else:
    writer = SummaryWriter(
        os.path.dirname(os.path.abspath(__file__))
        + "/runs/{}/{}_SAC_{}_{}".format(
            args.env_name,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.policy,
            "autotune" if args.automatic_entropy_tuning else "",
        )
    )

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = len(event_acc.Scalars("loss/policy")) if args.resume is True else 0
i_episode = len(event_acc.Scalars("reward/train")) if args.resume is True else 1
latest_avg_reward = (
    float(latest_file.split("_")[-1]) if args.resume is True else float("-inf")
)

for i_episode in itertools.count(i_episode):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps and not args.resume is True:
            # action = env.action_space.sample()  # Sample random action
            if args.policy == "Gaussian":
                action = (
                    torch.distributions.Uniform(-1, 1)
                    .sample((1, continuous_action_size))
                    .numpy()
                )
            else:
                action = (
                    torch.distributions.Uniform(-1, 1)
                    .sample((1, discrete_action_size))
                    .numpy()
                )
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                (
                    critic_1_loss,
                    critic_2_loss,
                    policy_loss,
                    ent_loss,
                    alpha,
                ) = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)
                writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                writer.add_scalar("entropy_temprature/alpha", alpha, updates)
                updates += 1

        if args.policy == "Gaussian":
            next_state, reward, done, max_episode_steps = env.step(None, action)  # Step
        else:
            next_state, reward, done, max_episode_steps = env.step(action, None)  # Step
        reward = reward[0]
        done = done[0]
        max_episode_steps = max_episode_steps[0]
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if max_episode_steps else float(not done)

        memory.push(
            state, action, reward, next_state, mask
        )  # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar("reward/train", episode_reward, i_episode)
    print(
        "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
            i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
        )
    )

    if i_episode % 20 == 0 and args.eval is True:
        avg_reward = 0.0
        episodes = 5
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                if args.policy == "Gaussian":
                    next_state, reward, done, _ = env.step(None, action)  # Step
                else:
                    next_state, reward, done, _ = env.step(action, None)  # Step
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar("avg_reward/test", avg_reward, i_episode)

        print("----------------------------------------")
        print(
            "Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2))
        )
        print("----------------------------------------")
        if latest_avg_reward < avg_reward or i_episode % 100 == 0:
            agent.save_checkpoint(args.env_name, avg_reward)
            if latest_avg_reward < avg_reward:
                latest_avg_reward = avg_reward

env.close()
