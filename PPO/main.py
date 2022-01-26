from tkinter import N
import gym
import numpy as np
from ppo import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    N = 20
    batch_size = 5
    n_epochs = 4
    lr = 0.0003

    agent = Agent(num_action=env.action_space.n,
                  input_dims=env.observation_space.shape,
                  batch_size=batch_size,
                  lr=lr,
                  num_epochs=n_epochs)

    n_games = 300
    figure_file = 'PPO/model/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_step = 0

    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            n_step += 1
            action, prob, val = agent.choose_action(obs)
            obs_next, reward, done, _ = env.step(action)
            score += reward
            agent.remember(obs, action, prob, val, reward, done)
            if n_step % N == 0:
                agent.learn()
                learn_iters += 1
            obs = obs_next

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_step', n_step, 'learn_step', learn_iters)

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
