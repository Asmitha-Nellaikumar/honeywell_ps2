import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=16, out_channels=8):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GCNConv(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        z = F.relu(self.encoder(x, edge_index))
        x_hat = self.decoder(z)
        return x_hat, z
