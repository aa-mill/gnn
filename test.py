import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # Use "add" aggregation
        
        # Define the node and message MLPs
        self.node_mlp = Linear(in_channels, out_channels)
        self.msg_mlp = Linear(2*in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # Construct messages from source to target nodes
        temp = torch.cat([x_i, x_j], dim=1)  # Concatenate source and destination node features
        return self.msg_mlp(temp)  # Apply a neural network to construct the messages

    def update(self, aggr_out, x):
        # Update node embeddings by applying a neural network
        new_embedding = self.node_mlp(x)
        
        # Combine the previous node embedding with the aggregated messages
        return new_embedding + aggr_out