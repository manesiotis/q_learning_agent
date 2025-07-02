# agent_qlearning.py

import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha              # learning rate
        self.gamma = gamma              # discount factor
        self.epsilon = epsilon          # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}               # Q-table (dict of state -> action-values)

    def _get_state_key(self, state):
        return tuple(np.round(state, 2))  # convert to tuple with reduced precision for hashing

    def act(self, state):
        state_key = self._get_state_key(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # explore
        return np.argmax(self.q_table.get(state_key, np.zeros(self.action_size)))  # exploit

    def learn(self, state, action, reward, next_state, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
