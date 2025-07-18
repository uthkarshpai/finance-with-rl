from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_model(env, total_timesteps=20000):
    vec_env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_rl_trader")
    return model

def load_model(env):
    from stable_baselines3 import PPO
    vec_env = DummyVecEnv([lambda: env])
    model = PPO.load("ppo_rl_trader", env=vec_env)
    return model