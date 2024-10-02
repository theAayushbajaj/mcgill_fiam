import pandas as pd
import pickle
import json
import warnings
warnings.filterwarnings('ignore')


def get_top_features(fi_estimates, n_features=100):
    """
    Extract top N features based on both MDA and MDI methods.
    
    Parameters:
    fi_estimates (dict): Dictionary containing feature importance estimates
    n_features (int): Number of top features to extract (default: 100)
    
    Returns:
    dict: A dictionary with 'MDA', 'MDI', and 'combined' top features
    """
    top_features = {}
    
    for method in ['MDA', 'MDI']:
        # Sort features by absolute importance (descending order)
        sorted_features = fi_estimates[method]['imp']['mean'].abs().sort_values(ascending=False)
        
        # Get top N features
        top_features[method] = sorted_features.head(n_features).index.tolist()
    
    # Get combined unique top features
    combined_features = list(set(top_features['MDA'] + top_features['MDI']))
    
    # If the combined list is longer than n_features, prioritize based on normalized importance
    if len(combined_features) > n_features:
        feature_importance = {}
        for feature in combined_features:
            mda_imp = abs(fi_estimates['MDA']['imp'].loc[feature, 'mean'])
            mdi_imp = abs(fi_estimates['MDI']['imp'].loc[feature, 'mean'])
            
            # Normalize importances
            mda_norm = mda_imp / fi_estimates['MDA']['imp']['mean'].abs().max()
            mdi_norm = mdi_imp / fi_estimates['MDI']['imp']['mean'].abs().max()
            
            # Average of normalized importances
            feature_importance[feature] = (mda_norm + mdi_norm) / 2
        
        combined_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        combined_features = [feature for feature, _ in combined_features[:n_features]]
    
    top_features['combined'] = combined_features
    
    return top_features

# load the fi_estimates from the pickle file
with open('./fi_estimates.pkl', 'rb') as f:
    fi_estimates = pickle.load(f)

print(type(fi_estimates))
print(type(fi_estimates['MDA']))
print(fi_estimates)
top_100_features = get_top_features(fi_estimates, n_features=100)
output_file = './top_100_features.json'

with open(output_file, 'w') as f:
    json.dump(top_100_features, f, indent=2)