import numpy as np
import pandas as pd

class TradingEnvironment:
    def __init__(self, prices, window_size=10, initial_balance=10000):
        # Αν το prices έχει μέθοδο reset_index (π.χ. pandas Series/DataFrame), κάνε reset_index
        if hasattr(prices, 'reset_index'):
            self.prices = prices.reset_index(drop=True)
        else:
            # Αν είναι numpy array, άφησέ το ως έχει
            self.prices = prices
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.action_space = [0, 1, 2]  # 0: Hold, 1: Buy, 2: Sell
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # number of shares held
        self.total_asset = self.balance
        self.trades = []
        return self._get_observation()

    def _get_observation(self):
        window = self.prices[self.current_step - self.window_size:self.current_step]
        obs = np.array(window).reshape(-1)  # flat vector
        obs = np.append(obs, [self.balance, self.position])
        return obs

    def step(self, action):
        price = self.prices[self.current_step]
        reward = 0

        # Execute action
        if action == 1:  # Buy
            if self.balance >= price:
                self.position += 1
                self.balance -= price
                self.trades.append(('Buy', price))
        elif action == 2:  # Sell
            if self.position > 0:
                self.position -= 1
                self.balance += price
                self.trades.append(('Sell', price))

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        self.total_asset = self.balance + self.position * price
        reward = self.total_asset - self.initial_balance  # cumulative profit
        obs = self._get_observation()

        return obs, reward, done

    def get_total_asset(self):
        return self.total_asset
