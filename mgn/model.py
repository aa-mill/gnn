import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter
from data import NodeType


def MLP(input_size, hidden_size, hidden_layers, output_size, layer_norm=True):
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU())
    for _ in range(hidden_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_size, output_size))
    if layer_norm:
        layers.append(nn.LayerNorm(output_size))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    def __init__(self, num_features, batch_size, max_accumulations=1e6, eps=1e-8):
        super(Normalizer, self).__init__()
        self.max_accumulations = max_accumulations/batch_size
        self.eps = torch.tensor(eps)
        self.register_buffer('count', torch.zeros(1))
        self.register_buffer('accumulations', torch.zeros(1))
        self.register_buffer('sum', torch.zeros(num_features))
        self.register_buffer('squares', torch.zeros(num_features))

    def forward(self, data, accumulate=True):
        if accumulate and self.accumulations < self.max_accumulations:
            self.accumulate(data.detach())
        normalized_data = (data - self.mean.detach())/self.std.detach()
        return normalized_data
    
    def inverse(self, normalized_data):
        return normalized_data*self.std.detach() + self.mean.detach()

    def accumulate(self, data):
        self.count += data.size(0)
        self.accumulations += 1
        self.sum += data.sum(0)
        self.squares += (data**2).sum(0)

    @property
    def mean(self):
        safe_count = torch.maximum(self.count, torch.tensor(1.0))
        return self.sum/safe_count

    @property
    def std(self):
        safe_count = torch.maximum(self.count, torch.tensor(1.0))
        var = (self.squares/safe_count) - self.mean**2
        std = torch.sqrt(torch.maximum(var, torch.tensor(0.0)))
        return torch.maximum(std, self.eps)
    

class EdgeUpdate(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super(EdgeUpdate, self).__init__()
        self.mlp = MLP(3*input_size, hidden_size, hidden_layers, output_size)

    def forward(self, src, dest, edge_attr, u, batch):
        return self.mlp(torch.cat([src, dest, edge_attr], dim=1))


class NodeUpdate(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super(NodeUpdate, self).__init__()
        self.mlp = MLP(2*input_size, hidden_size, hidden_layers, output_size)

    def forward(self, x, edge_index, edge_attr, u, batch):
        _, col = edge_index
        sum = scatter(edge_attr, col.to(torch.int64), dim=0, reduce='sum')
        return self.mlp(torch.cat([x, sum], dim=1))


class Encoder(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, hidden_layers, latent_size):
        super(Encoder, self).__init__()
        self.node_mlp = MLP(node_size, hidden_size, hidden_layers, latent_size)
        self.edge_mlp = MLP(edge_size, hidden_size, hidden_layers, latent_size)

    def forward(self, x, edge_attr):
        return self.node_mlp(x), self.edge_mlp(edge_attr)
    

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, hidden_layers, output_size):
        super(Decoder, self).__init__()
        self.node_mlp = MLP(latent_size, hidden_size, hidden_layers, output_size, layer_norm=False)

    def forward(self, x):
        return self.node_mlp(x)
    

class EncodeProcessDecode(nn.Module):
    def __init__(
            self, 
            node_size,
            edge_size, 
            hidden_size, 
            hidden_layers, 
            latent_size, 
            output_size, 
            graph_layers, 
            batch_size,
            normalize=True
        ):
        super(EncodeProcessDecode, self).__init__()
        # define architecture
        self.node_size = node_size
        self.edge_size = edge_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.latent_size = latent_size
        self.output_size = output_size
        self.graph_layers = graph_layers
        self.normalize = normalize
        
        # define normalizers
        self.node_normalizer = Normalizer(node_size, batch_size)
        self.edge_normalizer = Normalizer(edge_size, batch_size)
        self.output_normalizer = Normalizer(output_size, batch_size)

        # define modules
        self.Encoder = Encoder(
            self.node_size, 
            self.edge_size, 
            self.hidden_size, 
            self.hidden_layers, 
            self.latent_size
        )
        self.Processor = MetaLayer(
            EdgeUpdate(
                self.latent_size, 
                self.hidden_size, 
                self.hidden_layers, 
                self.latent_size
            ), 
            NodeUpdate(
                self.latent_size, 
                self.hidden_size, 
                self.hidden_layers, 
                self.latent_size
            )
        )
        self.Decoder = Decoder(
            self.latent_size, 
            self.hidden_size, 
            self.hidden_layers, 
            self.output_size
        )

    def forward(self, x, edge_index, edge_attr):
        if self.normalize:
            x = self.node_normalizer(x)
            edge_attr = self.edge_normalizer(edge_attr)
        x, edge_attr = self.Encoder(x, edge_attr)
        for _ in range(self.graph_layers):
            x_update, edge_attr_update, _ = self.Processor(x, edge_index, edge_attr)
            # add residual connections
            x += x_update
            edge_attr += edge_attr_update
        return self.Decoder(x)
    
    def loss(self, x, data, criterion):
        change = data.y - data.x[:, :2]
        if self.normalize:
            change = self.output_normalizer(change)
        mask = (data.node_type == NodeType.NORMAL) | \
            (data.node_type == NodeType.OUTFLOW)
        mask = mask.squeeze()
        return criterion(x[mask], change[mask])