# Project 10 - Reinforcement Learning for Trading (Q-Learning Agent)

This project demonstrates a basic Q-learning agent trained to make trading decisions (Buy, Sell, Hold) on a synthetic stock price time-series.

---

## 🔧 Project Structure

```
project10_rl_qlearning/
│
├── environment.py         # Custom trading environment
├── agent_qlearning.py     # Q-learning agent
├── run_q_learning.py      # Main training and evaluation script
├── plots/
│   └── q_learning_rewards.png
└── README.md              # Project overview and instructions
```

---

## 📈 Environment

The agent interacts with a simple stock market simulation defined in `TradingEnv`, where:

- **State**: previous `N` normalized prices (e.g., 10)
- **Actions**:
  - `0`: Hold
  - `1`: Buy
  - `2`: Sell
- **Reward**: change in portfolio value
- **Positioning**: Agent can hold only one position at a time (long or flat)

---

## 🤖 Q-Learning Agent

The agent uses tabular Q-learning with:

- `ε-greedy` exploration
- Discretized state space (by rounding)
- Hyperparameters:
  - α (learning rate) = 0.1
  - γ (discount factor) = 0.95
  - ε (exploration rate): decays from 1.0 to 0.01

Q-table is stored as a Python dictionary:  
```python
{ state_tuple: [Q_hold, Q_buy, Q_sell] }
```

---

## 🏃 Run Training

Make sure required packages are installed:

```bash
pip install matplotlib numpy
```

Then run:

```bash
python run_q_learning.py
```

---

## 📊 Output

The script generates a plot of total rewards over episodes:

- Saved at: `plots/q_learning_rewards.png`
- Shows agent's learning progress (increased rewards over time)

---

## ✅ Future Improvements

- Switch to Deep Q-Network (DQN) with function approximation
- Use real historical stock data
- Add transaction costs and slippage
- Track portfolio value over time

---

## 📚 References

- Sutton & Barto - *Reinforcement Learning: An Introduction*
- OpenAI Gym design philosophy
