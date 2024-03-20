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
    def __init__(self,):
        super(EdgeUpdate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2))

    def forward(self, src, dest, edge_attr, u, batch):
        input = (src - dest)/edge_attr.norm(dim=1, keepdim=True)**2*edge_attr
        coeffs = self.mlp(torch.cat([src - dest, 
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
                x[data.mask] = data.bcs
                loss = criterion(x, data.y)
                val_loss += loss.item()
        model.train()
        for data in train:
            data.to(device)
            optimizer.zero_grad()
            x, _, _ = model(data.x, data.edge_index, data.edge_attr)
            x[data.mask] = data.bcs
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
    model = MetaLayer(EdgeUpdate(), NodeUpdate()).to(device)
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
    # train(model, train_loader, val_loader, optimizer, criterion, device, epochs, track)
    
    # test the model
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    # select on of the test meshes to visualize
    mesh = test_data[0]
    model.eval()
    with torch.no_grad():
        data = mesh.mesh2Graph().to(device)
        x, _, _ = model(data.x, data.edge_index, data.edge_attr)
        x[data.mask] = data.bcs
        field = x.detach().cpu()[:, 0].reshape(mesh.X.shape)
    vmin = min(mesh.Fx.min(), field.min())
    vmax = max(mesh.Fx.max(), field.max())
    mesh.plotMesh(mesh.Fx, 'truth.png', vmin, vmax)
    mesh.plotMesh(field, 'pred.png', vmin, vmax)