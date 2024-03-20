import torch
from torch_geometric.nn import MetaLayer
from torch_geometric.loader import DataLoader
from torch_scatter import scatter, scatter_max, scatter_min
import numpy as np
import squares
import torch.nn as nn
import wandb
import pickle


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


def train(model, train, val, optimizer, criterion, device, epochs, track=False):
    """
    Trains model using provided training and validation DataLoaders.

    Args:
        model (torch.nn.Module): The model to be trained.
        train (torch.utils.data.DataLoader): The training data.
        val (torch.utils.data.DataLoader): The validation data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        device (torch.device): The device to be used for training.
        epochs (int): Number of training epochs.
        track (bool, optional): If true, logs training to wandb.
    """
    print(f'Training on {device}...')
    if track:
        wandb.init(project='gnn',
                   config={'epochs': epochs})
        wandb.watch(model, log='all', log_freq=10, log_graph=True)

    # standard training pipeline
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0  
        model.eval()  
        with torch.no_grad():  
            for data in val:
                data.to(device)
                x, _, _ = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(x, data.y)
                val_loss += loss.item()
        model.train()
        for data in train:
            data.to(device)
            optimizer.zero_grad()
            x, _, _ = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(x, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'{epoch + 1}: '
              f't{train_loss/len(train.dataset):.3f} '
              f'v{val_loss/len(val.dataset):.3f}')
        if track:
            wandb.log({'epoch': epoch + 1, 
                       'train_loss': train_loss/len(train.dataset), 
                       'val_loss': val_loss/len(val.dataset)})  

    if track:
        wandb.finish()

    # save the trained model
    torch.save(model.state_dict(), 'trained_model.pt')


def newDataset(train_size, val_size, test_size):
    """
    Creates new train, validation, and test datasets.

    Parameters:
        train_size (int): Number of training samples.
        val_size (int): Number of validation samples.
        test_size (int): Number of test samples.

    Returns:
        None
    """
    squares.createData(train_size, 'train_data.pkl')
    squares.createData(val_size, 'val_data.pkl')
    squares.createData(test_size, 'test_data.pkl', raw=True)

if __name__ == '__main__':
    # define meta parameters
    train_size = 1000
    val_size = 100
    test_size = 100
    epochs = 100
    track = False

    # create model
    model = GNN(
        nc=1, # node channels
        ec=2, # edge channels
        hc=32, # hidden channels
        ic=16, # intermediate channels
        oc=2 # output channels
        ).to(device)
    model.load_state_dict(torch.load('trained_model.pt'))

    # # create new dataset if needed
    # newDataset(train_size, val_size, test_size)

    # # load data
    # with open('train_data.pkl', 'rb') as f:
    #     train_data = pickle.load(f)
    # with open('val_data.pkl', 'rb') as f:
    #     val_data = pickle.load(f)
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # # define loss and optimizer
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # train the model
    # train(model, 
    #       train_loader, 
    #       val_loader,
    #       optimizer, 
    #       criterion, 
    #       device, 
    #       epochs, 
    #       track)
    
    # test the model
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    # select on of the test meshes to visualize
    mesh = test_data[11]
    model.eval()
    with torch.no_grad():
        data = mesh.mesh2Graph().to(device)
        x, _, _ = model(data.x, data.edge_index, data.edge_attr)
        field = x.detach().cpu()[:, 0].reshape(mesh.X.shape)
    mesh.plotMesh(mesh.Fx, 'truth.png')
    mesh.plotMesh(field, 'pred.png')