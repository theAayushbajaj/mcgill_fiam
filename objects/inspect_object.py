    #%%
import numpy as np
import pandas as pd
import pickle

#%%

# load objects/weights.pkl
with open("signals.pkl", "rb") as f:
    signals = pickle.load(f)
    

# %%

w = weights.iloc[-5]

# %%

weights
# %%
<<<<<<< HEAD

# load objects/mkt_ind.csv
benchmark_df = pd.read_csv("mkt_ind.csv")
benchmark_df
# %%

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# # Load data
# benchmark_df = pd.read_csv("mkt_ind.csv")

# # Calculate rolling volatility (e.g., 12-month rolling window)
# benchmark_df['volatility'] = benchmark_df['sp_ret'].rolling(window=12).std()

# # Drop NaN values after calculating rolling volatility
# volatility_data = benchmark_df['volatility'].dropna().values.reshape(-1, 1)

# # Fit HMM (let's start with 2 hidden states)
# hmm_model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=2000).fit(volatility_data)

# # Predict hidden states (market regimes)
# hidden_states = hmm_model.predict(volatility_data)

# # Create a figure and a set of subplots
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Plot the first dataset on the first y-axis
# ax1.plot(benchmark_df['t1'], benchmark_df['volatility'], label="Rolling Volatility", color='b')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Rolling Volatility', color='b')
# ax1.tick_params(axis='y', labelcolor='b')

# # Create a second y-axis sharing the same x-axis
# ax2 = ax1.twinx()
# ax2.plot(benchmark_df['t1'][len(benchmark_df['t1']) - len(hidden_states):], hidden_states, label="Hidden States", linestyle='--', color='r')
# ax2.set_ylabel('Hidden States', color='r')
# ax2.tick_params(axis='y', labelcolor='r')

# # Add legends
# fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))

# # Show the plot
# plt.show()


# %%

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

N_STATES = 2

# Load your dataset
benchmark_df = pd.read_csv("mkt_ind.csv")

# Assuming you're using EMA volatility as the observation
returns_or_volatility = benchmark_df["sp_ret"].rolling(window=12).std()

# Drop NaN values that appear due to the EMA calculation
returns_or_volatility = returns_or_volatility.dropna()

# Create a dataframe to store the inferred states
inferred_states = pd.Series(index=returns_or_volatility.index, dtype=int)

# Initialize an empty list to store the rolling volatility values for plotting
rolling_volatility_values = []

# Set initial training window
initial_training_window = 100

# Iteratively train the HMM and predict the state at each time t
for t in range(initial_training_window, len(returns_or_volatility)):
    # Use data up to time t-1 for training
    train_data = returns_or_volatility[:t].values.reshape(-1, 1)
    
    # Initialize and fit the HMM model on data up to time t-1
    hmm = GaussianHMM(n_components=N_STATES, covariance_type="diag", n_iter=1000)
    hmm.fit(train_data)
    
    # Predict the hidden state at time t using the trained HMM
    current_state = hmm.predict(returns_or_volatility[t-1:t].values.reshape(-1, 1))
    inferred_states.iloc[t] = current_state
    
    # Store the rolling volatility value for plotting
    rolling_volatility_values.append(returns_or_volatility.iloc[t])

# Create a figure and a set of subplots
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the first dataset on the first y-axis
ax1.plot(returns_or_volatility.index[initial_training_window:], rolling_volatility_values, label="Rolling Volatility", color='b')
ax1.set_xlabel('Date')
ax1.set_ylabel('Rolling Volatility', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(inferred_states.index[initial_training_window:], inferred_states.iloc[initial_training_window:], linestyle="--", color="orange", label="Hidden States")
ax2.set_ylabel('Hidden States', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add legends
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))

# Show the plot
plt.show()


# %%
=======
>>>>>>> 840ede1 (Let's go Moosa)
