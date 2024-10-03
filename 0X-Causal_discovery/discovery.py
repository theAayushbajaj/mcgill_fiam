"""
This script runs causal discovery on the given financial data to find correlated features.
"""
## Requires - pip install avici
## pip install networkx

import avici
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import networkx as nx

#from pgmpy.models import BayesianNetwork

model = avici.load_pretrained(download="scm-v0")

df = pd.read_pickle('../object/causal_dataset.pkl')
df = df.dropna()
x = df.to_numpy().astype(np.float32)

scaler = StandardScaler()
x= scaler.fit_transform(x)

g_prob = model(x=x)


THRESHOLD = 0.5

adj_matrix = (g_prob > THRESHOLD).astype(int)

G = nx.DiGraph(adj_matrix)

# map nodes to variable names
variable_names = df.columns.tolist()
mapping = {i: var_name for i, var_name in enumerate(variable_names)}
G = nx.relabel_nodes(G, mapping)

# perform d seperation

# model_pgmpy = BayesianNetwork()
# model_pgmpy.add_nodes_from(variable_names)
# edges = list(G.edges())
# model_pgmpy.add_edges_from(edges)

TGT_VAR = 'target'
variables = [var for var in variable_names if var != TGT_VAR]

for feature in variables:
    # Find all simple paths from the feature to the target
    paths = list(nx.all_simple_paths(G, source=feature, target=TGT_VAR))
    causal_paths = []
    backdoor_paths = []

    for path in paths:
        # Check the direction of edges along the path
        IS_CAUSAL = True
        for i in range(len(path) - 1):
            if not G.has_edge(path[i], path[i+1]):
                IS_CAUSAL = False
                break
        if IS_CAUSAL:
            causal_paths.append(path)
        else:
            backdoor_paths.append(path)

    if causal_paths:
        print(f"Feature '{feature}' has causal paths to \
              '{TGT_VAR}': {causal_paths}")

    if backdoor_paths:
        print(f"Feature '{feature}' has backdoor paths to \
              '{TGT_VAR}': {backdoor_paths}")

    if not causal_paths and backdoor_paths:
        print(f"Feature '{feature}' may have a spurious correlation with \
              '{TGT_VAR}' due to backdoor paths.")

    elif not causal_paths and not backdoor_paths:
        print(f"No paths found between '{feature}' and \
              '{TGT_VAR}'. They may be independent.")

#save g_prob
np.save('../objects/g_prob.npy', g_prob)

# for var in variables:
#     cond_set = [v for v in variables if v != var]
#     d_separated = not model_pgmpy.is_dconnected(TGT_VAR, var, observed=cond_set)
#     status = 'd-separated' if d_separated else 'd-connected'
#     print(f"Variables '{TGT_VAR}' and '{var}' are {status} given {cond_set}")


# # Compute and display graph metrics
# num_nodes = G.number_of_nodes()
# num_edges = G.number_of_edges()
# average_degree = sum(dict(G.degree()).values()) / num_nodes
# density = nx.density(G)
# print(f"Graph has {num_nodes} nodes, {num_edges} edges, \
# average degree {average_degree:.2f}, and density {density:.4f}")

# # Calculate the strongly connected components
# strongly_connected = list(nx.strongly_connected_components(G))
# print(f"Number of strongly connected components: {len(strongly_connected)}")

# # save graph
# nx.write_adjlist(G, "../objects/inferred_graph.adjlist")

# # Visualize the graph
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G, seed=42)
# nx.draw_networkx_nodes(G, pos, node_size=600, node_color='skyblue')
# nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
# nx.draw_networkx_labels(G, pos, font_size=10)
# plt.title("Inferred Causal Graph")
# plt.axis('off')
# plt.show()

# # perform topological sorting
# try:
#     topo_order = list(nx.topological_sort(G))
#     print(f"Topological order of nodes: {topo_order}")
# except nx.NetworkXUnfeasible:
#     print("Graph contains cycles, topological sort not possible.")


# # cluster the nodes using community detection
# from networkx.algorithms.community import greedy_modularity_communities

# communities = list(greedy_modularity_communities(G.to_undirected()))
# print(f"Detected {len(communities)} communities:")
# for i, community in enumerate(communities):
#     print(f"Community {i+1}: {list(community)}")
