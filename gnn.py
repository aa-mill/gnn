import torch
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter, scatter_max, scatter_min
import torch.nn as nn


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EdgeUpdate(nn.Module):
    """
    EdgeUpdate module for updating edge features.

    Args:
        nc (int): Number of input node features.
        ec (int): Number of input edge features.
        hc (int): Number of hidden units in MLP layers.
        ic (int): Number of output edge features.

    Attributes:
        mlp (nn.Sequential): MLP for edge feature transformation.
    """
    def __init__(self, nc, ec, hc, ic):
        super(EdgeUpdate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*nc + ec, hc),
            nn.ReLU(),
            nn.Linear(hc, hc),
            nn.ReLU(),
            nn.Linear(hc, ic)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        """
        Computes forward pass of the EdgeUpdate module.

        Args:
            src (torch.Tensor): Source node features.
            dest (torch.Tensor): Destination node features.
            edge_attr (torch.Tensor): Edge features.
            u (torch.Tensor): Global node features.
            batch (torch.Tensor): Batch indices.

        Returns:
            torch.Tensor: Updated edge features.
        """
        inputs = torch.cat([src, dest, edge_attr], dim=1)
        # keep original edge features and concatenate with updated edge features
        return torch.cat([src, dest, edge_attr, self.mlp(inputs)], dim=1)


class NodeUpdate(nn.Module):
    def __init__(self, nc, ec, hc, ic, oc):
        """
        NodeUpdate module for updating node features.

        Args:
            nc (int): Number of input node features.
            ec (int): Number of input edge features.
            hc (int): Number of hidden units in MLP layers.
            ic (int): Number of updated edge features.
            oc (int): Number of output node features.

        Attributes:
            mlp (nn.Sequential): MLP for node feature transformation.
        """
        super(NodeUpdate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nc + 5*(2*nc + ec + ic), hc), 
            nn.ReLU(), 
            nn.Linear(hc, hc), 
            nn.ReLU(), 
            nn.Linear(hc, oc))

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        Computes forward pass of NodeUpdate module.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indicies.
            edge_attr (torch.Tensor): Edge features.
            u (torch.Tensor): Global node features.
            batch (torch.Tensor): Batch indices.

        Returns:
            torch.Tensor: Updated node features.
        """
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
        """
        Initializes the meta GNN model, i.e. each layer is one message pass.

        Args:
            nc (int): Number of input node features.
            ec (int): Number of input edge features.
            hc (int): Number of hidden units in MLP layers.
            ic (int): Number of updated edge features.
            oc (int): Number of output node features.

        Attributes:
            input_layer (MetaLayer): Input layer of the GNN.
            output_layer (MetaLayer): Output layer of the GNN.
        """
        super(GNN, self).__init__()
        self.input_layer = MetaLayer(
            EdgeUpdate(nc=nc, ec=ec, hc=hc, ic=ic), 
            NodeUpdate(nc=nc, ec=ec, hc=hc, ic=ic, oc=oc))
        # self.output_layer = MetaLayer(
        #     EdgeUpdate(nc=4, ec=2*nc + ec + ic, hc=hc, ic=ic), 
        #     NodeUpdate(nc=4, ec=2*nc + ec + ic, hc=hc, ic=ic, oc=oc))

    def forward(self, x, edge_index, edge_attr):
        """
        Computes forward pass of the GNN model.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indicies.
            edge_attr (torch.Tensor): Edge features.

        Returns:
            torch.Tensor: Updated node features.
        """
        x, edge_attr, _ = self.input_layer(x, edge_index, edge_attr)
        # x, _, _ = self.output_layer(x, edge_index, edge_attr)
        return x, _, _


model = GNN(
    nc=1, # node channels
    ec=2, # edge channels
    hc=32, # hidden channels
    ic=16, # intermediate channels
    oc=2 # output channels
    ).to(device)