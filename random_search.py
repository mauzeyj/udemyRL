"""
For # of tme I want to adjust the weights
    new_weights = random
    For # episodes I want to play to decide whether to update the weights
        Play episode
    if avg episode length > best so far:
        weights = new_weights
Play a final set of episodes to see how good my best weights do again\
"""

import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')


def make_decision(num):
    """  Supposed to set up a np array?  or tensors?
    state.dot(params) > 0 -> do action 1
    state.dot(params) < 0 -> do action 0
    """
    if num > 0:
        return 1
    else:
        return 0

average_episode_length_best = 0

for n_weights in range(2000):
    model = np.random.random(4) * 2 - 1

    episode_list = []
    for episodes in range(50):
        steps = 0
        done = False
        state = env.reset()
        while done != True:
            prediction = np.dot(model, state)
            state, reward, done, _ = env.step(make_decision(prediction))
            steps += 1
        episode_list.append(steps)
        current_score = sum(episode_list) / len(episode_list)
        if current_score > average_episode_length_best:
            average_episode_length_best = current_score
            np.save('./models/random.npy', model)

# use best with env
best_model = np.load('./models/random.npy')
env = wrappers.Monitor(env, './data/')
for n in range(3):
    state = env.reset()
    while done != True:
        prediction = np.dot(model, state)
        state, reward, done, _ = env.step(make_decision(prediction))
        print(reward)
