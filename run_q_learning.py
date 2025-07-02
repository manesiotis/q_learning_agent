# run_q_learning.py

import numpy as np
import matplotlib.pyplot as plt
from environment import TradingEnvironment
from agent_qlearning import QLearningAgent

# === Load or generate data ===
# Dummy stock prices for simplicity (e.g., sinusoidal)
np.random.seed(42)
prices = 100 + np.cumsum(np.random.normal(0, 1, 1000))  # synthetic price series

# === Create environment and agent ===
env = TradingEnvironment(prices, window_size=10)
state_size = env.window_size
action_size = 3  # 0: Hold, 1: Buy, 2: Sell

agent = QLearningAgent(state_size, action_size)

# === Train agent ===
episodes = 10000
rewards_history = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    rewards_history.append(total_reward)
    print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}")

# === Plot rewards ===
plt.plot(rewards_history)
plt.title("Q-Learning Agent Performance")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/q_learning_rewards.png")
plt.show()
