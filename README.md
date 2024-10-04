# Mcgill_fiam

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
   python train_fixed_size_memory_optimized_pca.py
   cd ..
   ```

## Output

After running all the scripts, you'll find the following output:

- `objects/FULL_stacked_data.pkl`   
- `objects/causal_dataset.pkl`     
- `objects/predictions_1.csv to predictions_13.csv`   
- `objects/WEIGHT_SAMPLING.pkl`    
- `objects/mkt_ind.csv`            
- `objects/X_DATASET.pkl`                   
- `objects/Y_DATASET.pkl`          
- `objects/predictions_0.csv to predictions_13.csv`
- `objects/predictions.csv` -- Final predictions on the test dataset      

## Additional Information

For more details on each step of the process, please refer to the comments and docstrings within each script.


