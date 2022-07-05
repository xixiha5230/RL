import argparse
import os
from time import sleep
import gym
import glob
from Algorithm.sac import SAC

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
parser.add_argument(
    "--env-name",
    default="LunarLander-v2",
    help="Mujoco Gym environment (default: HalfCheetah-v2)",
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
    "--batch_size", type=int, default=256, metavar="N", help="batch size (default: 256)"
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
parser.add_argument("--cuda", action="store_true", help="run on CUDA (default: False)")
args = parser.parse_args()


env = gym.make(args.env_name, continuous=(args.policy == "Gaussian"))
agent = SAC(env.observation_space.shape[0], env.action_space, args)
list_of_files = glob.glob(
    os.path.dirname(os.path.abspath(__file__)) + "/../result/checkpoints/*"
)
latest_file = max(list_of_files, key=os.path.getctime)
print("Test on : ", latest_file)
latest_file = "result/checkpoints/sac_checkpoint_LunarLander-v2_268.7819370019341"
agent.load_checkpoint(latest_file)
done = False
obs = env.reset(seed=args.seed)
total_reward = 0
env.render()
while not done:
    action = agent.select_action(obs, False)
    obs, reward, done, _ = env.step(action)
    env.render()
    sleep(0.01)
    total_reward += reward
print("Reward: ", total_reward)