from data_loader import download_data
from environment import PortfolioEnv
from model import train_model
from evaluate import evaluate_model, calculate_metrics
import matplotlib.pyplot as plt

tickers = ['AAPL', 'MSFT', 'AMZN']
start_date = '2018-01-01'
end_date = '2020-12-31'

prices = download_data(tickers, start_date, end_date)
env = PortfolioEnv(prices)

model = train_model(env)
equity = evaluate_model(model, env)

metrics = calculate_metrics(equity)
print("Performance Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

plt.plot(equity, label='RL Portfolio Value')
plt.title("RL Portfolio Equity Curve")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.savefig("rl-portfolio-allocator/returns.png")
plt.show()
