#%%
import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from hrp import *

# Define HRP and HERC models as callable functions
def hrp_model(cov_matrix):
    sortIx = getQuasiDiag(sch.linkage(correlDist(cov_matrix.corr()), 'single'))
    return getRecBipart(cov_matrix, sortIx)

def herc_model(cov_matrix):
    sortIx = getQuasiDiag(sch.linkage(correlDist(cov_matrix.corr()), 'single'))
    return getRecBipart_HERC(cov_matrix, sortIx)

# Example usage with market data
market_data = pd.DataFrame(np.random.randn(1000, 5))  # 1000 days, 5 assets for demonstration

#%%

class PortfolioOptimizationEnv(gym.Env):
    def __init__(self, market_data, hrp_model, herc_model):
        super(PortfolioOptimizationEnv, self).__init__()
        
        # Market data for the environment
        self.market_data = market_data
        self.num_assets = market_data.shape[1]
        
        # HRP and HERC models
        self.hrp_model = hrp_model
        self.herc_model = herc_model
        
        # Action space: use Discrete for binary actions
        self.action_space = spaces.Discrete(2)  # 0 for HRP, 1 for HERC
        
        # Observation space: recent returns for each asset
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.done = False

    def reset(self):
        # Reset the environment to the initial state
        self.current_step = 1  # Start from 1 to avoid covariance issues
        self.done = False
        return self.market_data.iloc[self.current_step].values

    def step(self, action):
        # Execute one time step within the environment
        
        # Select the model (0 = HRP, 1 = HERC)
        if action == 0:
            cov_matrix = self.market_data.iloc[:self.current_step + 1].cov()
            weights = self.hrp_model(cov_matrix)
        elif action == 1:
            cov_matrix = self.market_data.iloc[:self.current_step + 1].cov()
            weights = self.herc_model(cov_matrix)
        else:
            raise ValueError("Invalid action.")
        
        # Handle potential NaN values in weights
        weights = weights.fillna(0).values
        
        # Ensure weights sum to 1
        weights /= np.sum(weights)
        
        # Calculate portfolio return
        portfolio_return = np.dot(weights, self.market_data.iloc[self.current_step].values)
        
        # Set reward based on portfolio return
        reward = portfolio_return
        
        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.market_data) - 1:
            self.done = True
        
        # Get the next state
        state = self.market_data.iloc[self.current_step].values
        
        return state, reward, self.done, {}

    def render(self, mode="human"):
        # Render the environment (optional)
        print(f"Step: {self.current_step}, Done: {self.done}")

#%%

# Create environment and wrap it in DummyVecEnv
env = DummyVecEnv([lambda: PortfolioOptimizationEnv(market_data, hrp_model, herc_model)])

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_portfolio_optimization")

# Load the trained model
model = PPO.load("ppo_portfolio_optimization")

# Run the environment with the trained model
obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()

print(f"Total Reward: {total_reward}")

# %%
