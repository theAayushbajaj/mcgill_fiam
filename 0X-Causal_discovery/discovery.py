# 0X-Causal_discovery/discovery.py

"""
This script runs causal discovery on the given financial data to find correlated features.
"""

import avici
import pandas as pd
import networkx as nx

model = avici.load_pretrained(download="scm-v0")

TGT_VAR = 'target'

# load top_100_features.json
top_100_features = pd.read_json('../0X-Causal_discovery/top_100_features.json')
top_100_features = top_100_features['combined'].to_list()
# pick the 50 first features
top_50_features = top_100_features[:50] + ['target'] # Due to compute constraints

x = pd.read_pickle('../objects/causal_dataset.pkl')


x = x[top_50_features]

df = x.copy()

x = x.to_numpy()
g_prob = model.predict(x)

THRESHOLD = 0.02

adj_matrix = (g_prob > THRESHOLD).astype(int)

G = nx.DiGraph(adj_matrix)

# map nodes to variable names
variable_names = df.columns.tolist()
mapping = {i: var_name for i, var_name in enumerate(variable_names)}
G = nx.relabel_nodes(G, mapping)

features = [var for var in variable_names if var != TGT_VAR]


# Assuming G is your causal DAG and features is your list of feature nodes

for feature in features:
    if nx.has_path(G, source=feature, target=TGT_VAR):
        print(f"There is a path from '{feature}' to '{TGT_VAR}'.")
    else:
        print(f"No path exists from '{feature}' to '{TGT_VAR}'.")
