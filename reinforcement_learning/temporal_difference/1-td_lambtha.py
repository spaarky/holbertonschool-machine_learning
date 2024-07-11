#!/usr/bin/env python3
"""Summary"""
import numpy as np


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Summary"""

    for _ in range(episodes):
        state = env.reset()
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            delta = reward + (gamma * V[next_state]) - V[state]

            eligibility *= lambtha * gamma
            eligibility[state] += 1

            V = V + alpha * delta * eligibility

            if done:
                break

            state = next_state

    return V
