import argparse
import torch
import gym

from Envwrapper.UnityEnv import UnityWrapper
from Module.MLPActorCritic import MLPActorCritic
from Algorithm.SAC import SAC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--exp_name", type=str, default="sac")
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())

    sac = SAC(
        lambda: gym.make(args.env, continuous=True),
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=None,
    )
    
    # sac = SAC(
    #     lambda: UnityWrapper(
    #         train_mode=True,
    #         file_name="venv_605/",
    #         no_graphics=False,
    #         seed=args.seed,
    #         work_id=1,
    #         continus=True,
    #     ),
    #     actor_critic=MLPActorCritic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
    #     gamma=args.gamma,
    #     seed=args.seed,
    #     epochs=args.epochs,
    #     replay_size=50000,
    #     start_steps=4000,
    #     update_after=400,
    #     num_test_episodes=3,
    #     steps_per_epoch=1000,
    #     logger_kwargs=None,
    # )
    sac.train()
