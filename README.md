# finance-rl-agent

This project trains a reinforcement learning (RL) agent to manage a stock portfolio using historical market data. The agent learns how to allocate capital across multiple assets to optimize long-term returns.

## Overview

- Uses daily adjusted closing prices from Yahoo Finance
- Trains a PPO agent in a custom OpenAI Gym environment
- Tracks portfolio value over time with real backtesting
- Outputs a performance plot (equity curve)

## What’s Inside

- `data_loader.py` – Downloads stock data and computes basic indicators
- `environment.py` – Custom trading environment for portfolio allocation
- `model.py` – PPO training logic with Stable Baselines3
- `evaluate.py` – Runs the trained agent and calculates Sharpe ratio & drawdown
- `main.py` – Orchestrates training, testing, and visualization

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train and evaluate
python main.py
