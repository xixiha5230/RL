import gym
import numpy
from numpy.lib.utils import info
from gridworld import CliffWalkingWapper
import time


class Agent():
    def __init__(self, obs_num, act_num, learning_rate, loss_rate,
                 export_rate):
        self.obs_num = obs_num
        self.act_num = act_num
        self.learning_rate = learning_rate
        self.loss_rate = loss_rate
        self.export_rate = export_rate
        self.Q = numpy.zeros((self.obs_num, self.act_num))

    def predict(self, obs):
        action_list_Q = self.Q[obs, :]
        max_Q = numpy.max(action_list_Q)
        action_list = numpy.where(action_list_Q == max_Q)[0]
        return numpy.random.choice(action_list)

    def sample(self, obs):
        if numpy.random.uniform(0, 1.0) < (1.0 - self.export_rate):
            action = self.predict(obs)
        else:
            action = numpy.random.choice(self.act_num)
        return action

    def learn(self, obs, action, reward, next_obs, next_action, done):
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.loss_rate * self.Q[next_obs, next_action]
        self.Q[obs, action] += self.learning_rate * (target_Q - predict_Q)

def train(env: CliffWalkingWapper, agent: Agent, is_render=False):
    obs = env.reset()
    action = agent.sample(obs)
    total_raward = 0
    step = 0
    while True:
        next_obs, reward, done, _ = env.step(action)
        next_action = agent.sample(next_obs)
        agent.learn(obs, action, reward, next_obs, next_action, done)
        total_raward += reward
        step += 1
        obs = next_obs
        action = next_action
        if is_render:
            env.render()
        if done:
            break
    print("step:", step, "total reward", total_raward)


def test(env, agent: Agent):
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        max_q = numpy.max(agent.Q[obs])
        action_list = numpy.where(agent.Q[obs] == max_q)[0]
        action = numpy.random.choice(action_list)
        next_obs, reward, done, info = env.step(action=action)
        obs = next_obs
        total_reward += reward
        env.render()
        time.sleep(0.1)
    print(total_reward)


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env=env)
    agent = Agent(env.observation_space.n, env.action_space.n, 0.1, 0.9, 0.1)
    for i in range(500):
        if i != 0 and i % 20 == 0:
            train(env, agent, True)
        else:
            train(env, agent, False)

    test(env, agent)
