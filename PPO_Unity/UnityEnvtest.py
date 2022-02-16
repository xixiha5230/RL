from gym import Space
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


def main():
    unity_env = UnityEnvironment("UnityEnv/CarDrive.x86_64")
    env = UnityToGymWrapper(unity_env, uint8_visual=True)
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            env.reset()
        print()


if __name__ == '__main__':

    main()
