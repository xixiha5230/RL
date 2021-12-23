import gym
import numpy
from numpy.lib.utils import info
from gridworld import CliffWalkingWapper
import time

env = gym.make("CliffWalking-v0")
env = CliffWalkingWapper(env=env)

Q_table = numpy.zeros(shape=(env.observation_space.n, env.action_space.n))

export_rate = 0.1
loss_rate = 0.9
learn_rate = 0.1


def sample(obs, env):
    def predict(obs):
        Q_list = Q_table[obs, :]
        maxQ = numpy.max(Q_list)
        action_list = numpy.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = numpy.random.choice(action_list)
        return action

    if numpy.random.uniform(0, 1) < (1.0 - export_rate):  #根据table的Q值选动作
        action = predict(obs)
    else:
        action = numpy.random.choice(env.action_space.n)  #有一定概率随机探索选取一个动作
    return action


def update_Q(obs, action, reward, next_obs, next_action, done):
    predict_Q = Q_table[obs, action]
    if done:
        target_Q = reward  # 没有下一个状态了
    else:
        target_Q = reward + loss_rate * Q_table[next_obs, next_action]  # Sarsa
    Q_table[obs, action] += learn_rate * (target_Q - predict_Q)  # 修正q


def train_step(total_step, env):
    obs = env.reset()
    action = sample(obs=obs, env=env)
    while True:
        next_obs, reward, done, info = env.step(action)

        next_action = sample(obs=next_obs, env=env)
        update_Q(obs, action, reward, next_obs, next_action, done)
        obs = next_obs
        action = next_action
        if total_step % 50 == 0 and total_step != 0:
            env.render()
        if done:
            break


def test(env):
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        max_q = numpy.max(Q_table[obs])
        action_list = numpy.where(Q_table[obs] == max_q)[0]
        action = numpy.random.choice(action_list)
        next_obs, reward, done, info = env.step(action=action)
        obs = next_obs
        total_reward += reward
        env.render()
        time.sleep(0.1)
    print(total_reward)


if __name__ == "__main__":
    for i in range(500):
        train_step(i, env)
        print(i)
    test(env)