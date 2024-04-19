#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import argparse
import os
import wandb
from model import EncodeProcessDecode
from tqdm import tqdm


# set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN.')
    parser.add_argument('--train_path', type=str, default='data/cylinder_flow',
                        help='Path to training data.')
    parser.add_argument('--val_path', type=str, default='data/cylinder_flow',
                        help='Path to validation data.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=0.999975,
                        help='Learning rate decay factor.')
    parser.add_argument('--track', action='store_true',
                        help='Track training with wandb.')
    return parser.parse_args()


def train(model, train, optimizer, criterion, device, epochs, batch_size, scheduler, track=False):
    # convert b/w steps and epochs
    log_at = int(1000/batch_size)
    save_at = int(100000/batch_size)
    warmup = int(1000/batch_size)

    # announce training has started
    print(f'Training on {device}...')
    if track:
        wandb.init(project='mgn',
                   config={'epochs': epochs, 
                           'batch_size': batch_size,
                           'lr': optimizer.param_groups[0]['lr'],
                           'gamma': scheduler.gamma})
        wandb.watch(model, log='all', log_freq=10, log_graph=True)

    # standard training pipeline
    step = 0
    for epoch in range(epochs):
        model.train()
        print(f'Starting epoch {epoch + 1}...')
        for data in tqdm(train):
            # move data to device
            data.to(device)

            # compute loss
            x = model(data.x, data.edge_index, data.edge_attr)
            loss = model.loss(x, data, criterion)

            # accumulate normalization statistics
            if step < warmup:
                with torch.no_grad():
                    # ensure statistics are actually computed
                    _ = loss
            else:
                # normal training
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                scheduler.step()
            step += 1

            # log training loss
            if step % log_at == 0 and step > warmup:
                if track:
                    wandb.log({'step': step,
                            'train_loss': train_loss})
                
            # save model checkpoint
            if step % save_at == 0:
                torch.save(model.state_dict(), 
                           f'chkpts/chkpt_{step}.pt')

    # finish tracking
    if track:
        wandb.finish()

    # save the trained model
    torch.save(model.state_dict(), 'chkpts/final.pt')


def main():
    args = parse_args()

    # create model
    model = EncodeProcessDecode(
        node_size=9, 
        edge_size=3, 
        hidden_size=128, 
        hidden_layers=2, 
        latent_size=128, 
        output_size=2,
        graph_layers=15,
        batch_size=args.batch_size,
        normalize=True
    ).to(device)

    # load data
    train_data = []
    with open(os.path.join(args.train_path, 'train.pt'), 'rb') as f:
        data = torch.load(f)
        for trajectory in data.values():
            train_data.extend(trajectory)  
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=8, shuffle=True)

    # define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    
    # train the model
    train(model, 
          train_loader, 
          optimizer, 
          criterion, 
          device, 
          args.epochs, 
          args.batch_size,
          scheduler, 
          track=True)
    print(f'Final learning rate: {scheduler.get_last_lr()[0]:.3e}')


if __name__ == '__main__':
    main()