import torch
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter, scatter_max, scatter_min
import torch.nn as nn


class EdgeUpdate(nn.Module):
    def __init__(self, nc, ec, hc, ic):
        super(EdgeUpdate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*nc + ec, hc),
            nn.ReLU(),
            nn.Linear(hc, hc),
            nn.ReLU(),
            nn.Linear(hc, ic)
        ).double()

    def forward(self, src, dest, edge_attr, u, batch):
        inputs = torch.cat([src, dest, edge_attr], dim=1)
        # keep original edge features and concatenate with updated edge features
        return torch.cat([src, dest, edge_attr, self.mlp(inputs)], dim=1)


class NodeUpdate(nn.Module):
    def __init__(self, nc, ec, hc, ic, oc):
        super(NodeUpdate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nc + 5*(2*nc + ec + ic), hc), 
            nn.ReLU(), 
            nn.Linear(hc, hc), 
            nn.ReLU(), 
            nn.Linear(hc, oc)).double()

    def forward(self, x, edge_index, edge_attr, u, batch):
        _, dest = edge_index
        _, max_indices = scatter_max(edge_attr[:, 0], dest, dim=0)
        _, min_indices = scatter_min(edge_attr[:, 0], dest, dim=0)
        # features associated with max/min source node for each destination node
        max_edge_attr = edge_attr[max_indices]
        min_edge_attr = edge_attr[min_indices]
        # features mixed across edges via min/mean/max
        min = scatter(edge_attr, dest, dim=0, reduce='min')
        mean = scatter(edge_attr, dest, dim=0, reduce='mean')
        max = scatter(edge_attr, dest, dim=0, reduce='max')
        # inputs are current node features and aggregated edge features
        inputs = torch.cat([x, min_edge_attr, max_edge_attr, min, mean, max], dim=1)
        return self.mlp(inputs)


class GNN(nn.Module):
    def __init__(self, nc, ec, hc, ic, oc):
        super(GNN, self).__init__()
        self.input_layer = MetaLayer(
            EdgeUpdate(nc=nc, ec=ec, hc=hc, ic=ic), 
            NodeUpdate(nc=nc, ec=ec, hc=hc, ic=ic, oc=4))
        self.output_layer = MetaLayer(
            EdgeUpdate(nc=4, ec=2*nc + ec + ic, hc=hc, ic=ic), 
            NodeUpdate(nc=4, ec=2*nc + ec + ic, hc=hc, ic=ic, oc=oc))

    def forward(self, x, edge_index, edge_attr):
        x, edge_attr, _ = self.input_layer(x, edge_index, edge_attr)
        x, _, _ = self.output_layer(x, edge_index, edge_attr)
        return x, _, _


model = GNN(
    nc=2, # node channels
    ec=2, # edge channels
    hc=32, # hidden channels
    ic=16, # intermediate channels
    oc=2 # output channels
    )