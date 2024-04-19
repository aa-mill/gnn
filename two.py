import torch
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter
import torch.nn as nn


class EdgeUpdate(nn.Module):
    def __init__(self):
        super(EdgeUpdate, self).__init__()
        self.weight_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)).double()
        
        self.mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 5)).double()

    def forward(self, src, dest, edge_attr, u, batch):
        # unpacking features
        fj, fi = src[:, 0, None], dest[:, 0, None] # field values
        degree = dest[:, 1, None] # degree of destination node
        dx, dy = edge_attr[:, 0, None], edge_attr[:, 1, None] # edge attributes
        angle = torch.atan2(dy, dx) # angle of edge
        distance = edge_attr.norm(dim=1, keepdim=True) # node separation
        dd = (fj - fi)/distance**2*edge_attr # directional derivative

        # mlp predicts weights for each directional derivative
        inputs = torch.cat([degree, angle, distance], dim=1)
        weights = self.weight_mlp(inputs)

        # mlp generates hidden features
        inputs = torch.cat([degree, angle, distance, dd], dim=1)
        hidden = self.mlp(inputs)
        return torch.cat([dd*weights, hidden], dim=1)
    

class NodeUpdate(nn.Module):
    def __init__(self):
        super(NodeUpdate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)).double()

    def forward(self, x, edge_index, edge_attr, u, batch):
        # unpack features
        dds = edge_attr[:, :2]
        hidden = edge_attr[:, 2:]

        # linear combination of directional derivatives
        _, col = edge_index
        dd = scatter(dds, col, dim=0, reduce='sum')

        # aggregate hidden features
        sum = scatter(hidden, col, dim=0, reduce='sum')
        min = scatter(hidden, col, dim=0, reduce='min')
        max = scatter(hidden, col, dim=0, reduce='max')

        inputs = torch.cat([dd, sum, min, max], dim=1)
        return self.mlp(inputs)


model = MetaLayer(EdgeUpdate(), NodeUpdate())