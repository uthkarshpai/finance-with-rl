import gym
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, prices, window=30, initial_cash=1e6):
        self.prices = prices
        self.window = window
        self.initial_cash = initial_cash
        self.n_assets = prices.shape[1]
        self.reset()

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(window, self.n_assets), dtype=np.float32)

    def reset(self):
        self.t = self.window
        self.cash = self.initial_cash
        self.history = [self.cash]
        return self._get_obs()

    def _get_obs(self):
        window_prices = self.prices.iloc[self.t - self.window:self.t].values
        return window_prices / window_prices[0]  # Normalize to first row

    def step(self, action):
        if np.sum(action) == 0:
            action = np.ones_like(action)
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights)

        prev_value = self.cash
        returns = self.prices.iloc[self.t].values / self.prices.iloc[self.t - 1].values
        self.cash *= np.dot(weights, returns)
        self.history.append(self.cash)
        self.t += 1

        done = self.t >= len(self.prices) - 1
        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        reward = (self.cash - prev_value) / prev_value  # Optional: add risk penalty
        return obs, reward, done, {"cash": self.cash}
