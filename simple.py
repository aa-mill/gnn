import torch
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter
import torch.nn as nn


class EdgeUpdate(nn.Module):
    def __init__(self,):
        super(EdgeUpdate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).double()

    def forward(self, src, dest, edge_attr, u, batch):
        fj, fi = src[:, 0, None], dest[:, 0, None] # field values
        dx = edge_attr[:, 0, None]
        # degree = dest[:, 1, None] # degree of destination node
        hops = edge_attr[:, 1, None] # hops from src to dest
        dd = (fj - fi)/dx # directional derivative
        weights = self.mlp(torch.cat([torch.abs(hops)], dim=1))
        return torch.cat([dd*weights, weights], dim=1)       


class NodeUpdate(nn.Module):
    def __init__(self):
        super(NodeUpdate, self).__init__()

    def forward(self, x, edge_index, edge_attr, u, batch):
        _, col = edge_index
        # compute linear combination of edge updates
        sum = scatter(edge_attr[:, :-1], col, dim=0, reduce='sum')
        return sum, edge_attr[:, -1]


model = MetaLayer(EdgeUpdate(), NodeUpdate())