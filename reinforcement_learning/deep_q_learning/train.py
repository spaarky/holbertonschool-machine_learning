#!/usr/bin/env python3
"""Summary"""

import cv2
import gym
import numpy as np
from gym.core import ObservationWrapper
from gym.spaces import Box
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam


class PreprocessAtari(ObservationWrapper):
    """Summary"""
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)

        self.img_size = (84, 84)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0],
                                     self.img_size[1], 1))

    def observation(self, img):
        """Summary"""

        img = img[34:-16, :, :]
        img = cv2.resize(img, self.img_size)
        img = img = img.mean(-1, keepdims=True)
        img = img.astype('float32') / 255.

        return img


def build_model(height, width, channels, actions):
    """Summary"""
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu',
                            input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    """Summary"""
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, policy=policy, memory=memory,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=100000,
                   target_model_update=1e-2)
    return dqn


def train():
    """Summary"""
    env = gym.make("Breakout-v0")
    env = PreprocessAtari(env)

    height, width, channels = env.observation_space.shape
    actions = env.action_space.n
    # env.unwrapped.get_action_meanings()
    model = build_model(height, width, channels, actions)
    # model.summary()

    dqn = build_agent(model, actions)
    dqn.compile(Adam(learning_rate=1e-4), metrics=['mae'])

    dqn.fit(env, nb_steps=600000, visualize=False, verbose=2)

    try:
        dqn.save_weights('policy.h5')
        print('Saved')
    except Exception as e:
        print('Not Saved')
        print(e)

    return dqn


if __name__ == "__main__":
    dqn = train()

    env = gym.make("Breakout-v0")
    env = PreprocessAtari(env)
    scores = dqn.test(env, nb_episodes=10, visualize=False)
    print(np.mean(scores.history['episode_reward']))
