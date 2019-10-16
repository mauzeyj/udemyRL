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
import tensorflow as tf

env = gym.make('CartPole-v0')

# done = False
# while not done:
#     observation, reward, done, info = env.step(env.action_space.sample())

"""  what is going on here
state.dot(params) > 0 -> do action 1
state.dot(params) < 0 -> do action 0
"""
average_episode_length_best = 0

for n_weights in range(20):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_shape=(4,),
                                    kernal_initializer='random_uniform',
                                    bias_initializer='random_uniform',
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(4,
                                    kernal_initializer='random_uniform',
                                    bias_initializer='random_uniform',
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(2,
                                    kernal_initializer='random_uniform',
                                    bias_initializer='random_uniform',
                                    activation='softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    episodes = []
    for episodes in range(20):
        steps = 0
        done = False
        state = env.reset()
        while done == False:
            prediction = model.predict(state)
