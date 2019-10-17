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
import tensorflow as tf

env = gym.make('CartPole-v0')

# done = False
# while not done:
#     observation, reward, done, info = env.step(env.action_space.sample())

"""  Supposed to set up a np array?  or tensors?  Going to use Keras
state.dot(params) > 0 -> do action 1
state.dot(params) < 0 -> do action 0
"""
average_episode_length_best = 0

for n_weights in range(20):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_shape=(4,),
                                    # kernal_initializer='random_uniform',
                                    # bias_initializer='random_uniform',
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(4,
                                    # kernal_initializer='random_uniform',
                                    # bias_initializer='random_uniform',
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(2,
                                    # kernal_initializer='random_uniform',
                                    # bias_initializer='random_uniform',
                                    activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    episode_list = []
    for episodes in range(20):
        steps = 0
        done = False
        state = env.reset()
        steps = 0
        while done != True:
            prediction = model.predict(state.reshape(-1, 4))
            new_state, reward, done, _ = env.step(np.argmax(prediction))
            steps = + 1
        episode_list.append(steps)
    current_score = sum(episode_list) / len(episode_list)
    if current_score > average_episode_length_best:
        average_episode_length_best = current_score
        model.save('./models/random best.h5')

# use best with env
best_model = tf.keras.models.load_model('./models/random best.h5')
for n in range(3):
    test_state = env.reset()
    while done != True:
        env.render()
        test_state, reward, done, _ = env.step(best_model.predict(test_state).reshape(-1, 4))
        print(reward)
