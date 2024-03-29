import torch
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter
import torch.nn as nn


# set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class EdgeUpdate(nn.Module):
    def __init__(self,):
        super(EdgeUpdate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2))

    def forward(self, src, dest, edge_attr, u, batch):
        fj, fi = src[:, 0, None], dest[:, 0, None]
        degree = dest[:, 1, None]
        input = (fj - fi)/edge_attr.norm(dim=1, keepdim=True)**2*edge_attr
        coeffs = self.mlp(torch.cat([fj - fi,
                                     degree, 
                                     edge_attr, 
                                     edge_attr.norm(dim=1, keepdim=True)], dim=1))
        return input*coeffs


class NodeUpdate(nn.Module):
    def __init__(self):
        super(NodeUpdate, self).__init__()

    def forward(self, x, edge_index, edge_attr, u, batch):
        _, col = edge_index
        sum = scatter(edge_attr, col, dim=0, reduce='sum')
        return sum


model = MetaLayer(EdgeUpdate(), NodeUpdate())