"""
Bin the state in cartpole so that q table can be used.  
"""

import gym
import numpy as np


def take_action(s, w):
    return np.where(s.dot(w) > 0, 1, 0)


env = gym.make('CartPole-v0')

np.random.random(4) * 2 - 1
