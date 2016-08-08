#!/usr/bin/env python

import argparse
import gym
import numpy as np


def run_episode(env, weights, time_limit=2000, submit=False):
    observation = env.reset()
    episode_reward = 0
    for t in range(time_limit):
        if not submit:
            env.render()
        prediction = np.matmul(weights, observation)
        action = 0 if np.sign(np.dot(weights, observation)) <= 0 else 1
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            break
    print('Episode reward: {}'.format(episode_reward))
    return episode_reward


def train(env, episodes_limit=3000, step_factor=.5, submit=False):
    num_spaces = 4
    weights = np.random.rand(num_spaces) * 5
    best_result = 0

    # Train our linear model.
    for _ in range(episodes_limit):
        noise = np.random.rand(num_spaces) * step_factor
        result = run_episode(env, weights + noise, submit=submit)
        if result > best_result:
            weights, best_result = weights + noise, result

    return best_result, weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true', help='submit results')
    args = parser.parse_args()
    env = gym.make('CartPole-v0')
    print('Training our model')
    best_result, weights = train(env, submit=args.submit)
    if args.submit:
        env.monitor.start('cartpole-experiment/', force=True)
    print(best_result)
    if args.submit:
        print('Running tries for submission')
        submission_tries = 100
        for _ in range(submission_tries):
            run_episode(env, weights, submit=args.submit)
        env.monitor.close()
        gym.upload('cartpole-experiment/', api_key='sk_XJqn2jHQ5SAp2XuqoAgew')
