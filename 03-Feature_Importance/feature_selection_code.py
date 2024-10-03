"""
Feature Importance Extraction and Storage

This script is designed to load feature importance estimates from a pickle file,
extract the top features based on two methods (Mean Decrease Accuracy - MDA and
Mean Decrease Impurity - MDI), and save the results to a JSON file.

The extraction process involves the following steps:
1. Load the feature importance estimates from a serialized pickle file.
2. Identify the top N features based on their importance scores from both MDA and MDI methods.
3. Combine the lists of top features from both methods, ensuring uniqueness and
   prioritizing features based on their average normalized importance if necessary.
4. Save the resulting top features into a JSON file for further analysis or visualization.

"""

import pickle
import json
import warnings
warnings.filterwarnings('ignore')


def get_top_features(imp_estimates, n_features=100):
    """
    Extract top N features based on both MDA and MDI methods.

    Parameters:
    imp_estimates (dict): Dictionary containing feature importance estimates
    n_features (int): Number of top features to extract (default: 100)

    Returns:
    dict: A dictionary with 'MDA', 'MDI', and 'combined' top features
    """
    top_features = {}

    for method in ['MDA', 'MDI']:
        # Sort features by absolute importance (descending order)
        sorted_features = imp_estimates[method]['imp']['mean'].abs().sort_values(ascending=False)

        # Get top N features
        top_features[method] = sorted_features.head(n_features).index.tolist()

    # Get combined unique top features
    combined_features = list(set(top_features['MDA'] + top_features['MDI']))

    # If the combined list is longer than n_features, prioritize based on normalized importance
    if len(combined_features) > n_features:
        feature_importance = {}
        for feature in combined_features:
            mda_imp = abs(imp_estimates['MDA']['imp'].loc[feature, 'mean'])
            mdi_imp = abs(imp_estimates['MDI']['imp'].loc[feature, 'mean'])

            # Normalize importances
            mda_norm = mda_imp / imp_estimates['MDA']['imp']['mean'].abs().max()
            mdi_norm = mdi_imp / imp_estimates['MDI']['imp']['mean'].abs().max()

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
OUTPUT_FILE = './top_100_features.json'

with open(OUTPUT_FILE, 'w') as f:
    json.dump(top_100_features, f, indent=2)
