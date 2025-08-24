# src/build_graph.py
import numpy as np
import torch
from torch_geometric.utils import to_undirected

def build_graph(data, threshold=0.5):
    """
    Builds a correlation graph from the data.
    If no edges meet the threshold, it creates a graph with self-loops.
    """
    corr = np.corrcoef(data.T)
    
    edges = []
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i != j and abs(corr[i, j]) > threshold:
                edges.append([i, j])

    if not edges:
        # If no edges found, create a simple graph with self-loops
        num_features = data.shape[1]
        edge_index = torch.arange(num_features, dtype=torch.long).view(1, -1)
        edge_index = torch.cat([edge_index, edge_index], dim=0)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Convert to undirected graph
        edge_index = to_undirected(edge_index)
        
    return edge_index
