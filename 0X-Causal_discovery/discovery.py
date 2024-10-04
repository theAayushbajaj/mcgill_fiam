import avici
import pandas as pd
import networkx as nx

model = avici.load_pretrained(download="scm-v0")

# load top_100_features.json
top_100_features = pd.read_json('../0X-Causal_discovery/top_100_features.json')
top_100_features = top_100_features['combined'].to_list()
# pick the 50 first features
top_50_features = top_100_features[:50] + ['target']

x = pd.read_pickle('../objects/causal_dataset.pkl')


x = x[top_50_features]

df = x.copy()

x = x.to_numpy()
g_prob = model.predict(x)

threshold = 0.02

adj_matrix = (g_prob > threshold).astype(int)

G = nx.DiGraph(adj_matrix)

# map nodes to variable names
variable_names = x.columns.tolist()
mapping = {i: var_name for i, var_name in enumerate(variable_names)}
G = nx.relabel_nodes(G, mapping)

target_variable = 'target'
features = [var for var in variable_names if var != target_variable]


# Assuming G is your causal DAG and features is your list of feature nodes
target_variable = 'target'

for feature in features:
    if nx.has_path(G, source=feature, target=target_variable):
        print(f"There is a path from '{feature}' to '{target_variable}'.")
    else:
        print(f"No path exists from '{feature}' to '{target_variable}'.")