#!/usr/bin/env python3
"""Summary"""
import numpy as np


def epsilon_greedy(state, Q, epsilon):
    """Summary"""
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, int(Q.shape[1]))
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Summary"""
    epsilon_init = epsilon

    for ep in range(episodes):
        state = env.reset()
        action = epsilon_greedy(state, Q, epsilon)
        eligibility = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(next_state, Q, epsilon)

            delta = (reward + (gamma * Q[next_state, next_action])
                     - Q[state, action])

            eligibility *= lambtha * gamma
            eligibility[state, action] += 1

            Q += alpha * delta * eligibility

            if done:
                break

            state = next_state
            action = next_action

        epsilon = min_epsilon + (epsilon_init - min_epsilon) *\
            np.exp(-epsilon_decay * ep)

    return Q
