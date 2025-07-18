import numpy as np

def evaluate_model(model, env):
    obs = env.reset()
    done = False
    values = [env.cash]

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        values.append(info['cash'])

    return values

def calculate_metrics(equity_curve):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    drawdowns = 1 - np.array(equity_curve) / np.maximum.accumulate(equity_curve)
    max_dd = np.max(drawdowns)
    return {"Sharpe Ratio": sharpe, "Max Drawdown": max_dd}
