#!/usr/bin/env python3
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import wandb
import pickle
import simple, medium, two, gnn
import argparse
from total import EncodeProcessDecode


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train, val, optimizer, criterion, device, epochs, scheduler, track=False):
    """
    Trains model using provided training and validation DataLoaders.

    Args:
        model (torch.nn.Module): Model to be trained.
        train (torch.utils.data.DataLoader): Training data.
        val (torch.utils.data.DataLoader): Validation data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        criterion (torch.nn.Module): Loss function used for training.
        device (torch.device): Device to be used for training.
        epochs (int): Number of training epochs.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
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
        scheduler.step()
        # for data in train:
        #     # move data to device
        #     data.to(device)

        #     # compute loss
        #     x = model(data.x, data.edge_index, data.edge_attr)
        #     x[data.mask] = model.data.bcs
        #     y = model.output_normalizer(data.y)
        #     loss = criterion(x, y)

        #     # accumulate normalization statistics
        #     if epoch < 5:
        #         with torch.no_grad():
        #             # ensure statistics are actually computed
        #             _ = loss
        #     else:
        #         # normal training
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         train_loss = loss.item()
        #         scheduler.step()
        print(f'{epoch + 1}: '
              f'\tt{train_loss/len(train):.5e} '
              f'v{val_loss/len(val):.5e}')
        if track:
            wandb.log({'epoch': epoch + 1, 
                       'train_loss': train_loss/len(train), 
                       'val_loss': val_loss/len(val)})  

    if track:
        wandb.finish()

    # save the trained model
    torch.save(model.state_dict(), 'trained_model.pt')


def parse_args():
    """
    Parses command line arguments for generating graded block mesh.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train GNN.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['simple', 'medium', 'two', 'gnn'],
                        help='Model to train.')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate.')
    parser.add_argument('--track', action='store_true', 
                        help='If true, logs training to wandb.')
    parser.add_argument('--train_path', type=str, default='data/train_data.pkl',
                        help='Path to training data.')
    parser.add_argument('--val_path', type=str, default='data/val_data.pkl',
                        help='Path to validation data.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Exponential learning rate decay.')
    return parser.parse_args()


def main():
    args = parse_args()

    # create model
    model = getattr(globals()[args.model], 'model').to(device)

    # model = EncodeProcessDecode(
    #     node_size=2, 
    #     edge_size=3, 
    #     hidden_size=32, 
    #     hidden_layers=2, 
    #     latent_size=32, 
    #     output_size=2,
    #     graph_layers=5,
    #     batch_size=64,
    #     normalize=True
    # ).to(device)

    # load data
    with open(args.train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(args.val_path, 'rb') as f:
        val_data = pickle.load(f)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    # train the model
    train(model, 
          train_loader, 
          val_loader, 
          optimizer, 
          criterion, 
          device, 
          args.epochs, 
          scheduler, 
          args.track)
    print(f'Final learning rate: {scheduler.get_last_lr()[0]:.3e}')


if __name__ == '__main__':
    main()