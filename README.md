# Mcgill_fiam

Slides presented at McGill-FIAM: http://bit.ly/40GVIlI

# Instructions to run the code

This repository contains a series of scripts for data preprocessing, feature engineering, and feature importance analysis.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
2. Create a new directory `raw_data` and add the following files to it:
   - `factor_char_list.csv`
   - `hackathon_sample_v2.csv`
   - `mkt_ind.csv`

## Running the Analysis

Follow these steps to run the complete analysis:

1. Data Preprocessing:
   ```
   cd 01-Data_Preprocessing
   python preprocessing_code.py
   cd ..
   ```

2. Feature Engineering:
   ```
   cd 02-Feature_Engineering
   python feature_engineering_code.py
   cd ..
   ```

3. Feature Importance:
   ```
   cd 03-Feature_Importance
   python feature_importance_code.py
   python feature_selection_code.py
   cd ..
   ```
4. Causal Discovery:
   ```
   cd 0X-Causal_discovery   
   python discovery.py
   cd ..
   ```
5. Predictor:
   ```
   cd 04-Predictor
   python train_AlphaSignals.py
   cd ..
   ```

6. Backtesting
   ```
   cd 06-Backtesting
   python backtest_parallel.py
   cd ..
   ```

7. Chain of Thought Zero Shot Features
   Download the dataset from [here](https://drive.google.com/drive/folders/1tZP9A0hrAj8ptNP3VE9weYZ3WDn9jHic) and put it in the `datasets` directory.
   Login into Hugging Face and download the model [here](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
   ```
   cd 0L-CoTZeroShotFeatures
   python create_dataset.py
   python llama-3.2-3B-Instruct-Inference-READABILITY-SCORE.py
   python llama-3.2-3B-Instruct-Inference-RISK_FACTORS.py
   python llama-3.2-3B-Instruct-Inference-SENTIMENT-SCORES.py
   cd ..
   ```
## Output

After running all the scripts, you'll find the following output:

- `objects/FULL_stacked_data.pkl`   
- `objects/causal_dataset.pkl`      
- `objects/WEIGHT_SAMPLING.pkl`    
- `objects/mkt_ind.csv`            
- `objects/X_DATASET.pkl`                   
- `objects/Y_DATASET.pkl`          
- `objects/predictions_0.csv to predictions_13.csv`
- `objects/predictions.csv` -- Final predictions on the test dataset
- `objects/prices.pkl` -- Dataframe of the prices of the assets
- `objects/signals.pkl` -- Timeseries of signals from each stock
- `objects/market_caps.pkl` -- Dataframe of market caps of the assets
- `objects/stock_exret.pkl` -- Excess returns of the each stock

<!-- The feature importance scores from MDI and MDA are: -->


## Additional Information

For more details on each step of the process, please refer to the comments and docstrings within each script.


