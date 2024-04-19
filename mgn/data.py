#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import h5py
import json
import os
import enum
from tqdm import tqdm as tdqm


class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6


def add_noise(data, noise_scale=0.02, noise_gamma=1):
    mask = data.node_type == NodeType.NORMAL
    noise = torch.randn_like(data.x[:, :2])*noise_scale*mask
    data.x[:, :2] += noise
    data.y += (1.0 - noise_gamma)*noise
    return


def to_pyg(load_dir, save_dir, split, noise=False):

    # load metadata
    with open(os.path.join(load_dir, 'meta.json'), 'r') as file:
        meta = json.loads(file.read())

    data = {}
    with h5py.File(os.path.join(load_dir, split + '.h5'), 'r') as file:
        # loop over all trajectories
        sequence = []
        for group in tdqm(file):
            # load one trajectory
            traj = file[group]

            # compute graph structure outside of time loop
            all_edges = []
            cells = traj['cells'][0]
            for tri in cells:
                edges = [(min(tri[i], tri[(i + 1) % 3]), 
                          max(tri[i], tri[(i + 1) % 3])) 
                          for i in range(3)]
                all_edges.extend(edges)
            edge_index = np.array(list(set(all_edges)), dtype=int).T

            # compute edge attributes, which do not change either
            mesh_pos = traj['mesh_pos'][0]
            edge_attr = []
            for src, dest in edge_index.T:
                dx, dy = mesh_pos[src] - mesh_pos[dest]
                edge_attr.append([dx, dy, np.sqrt(dx**2 + dy**2)])
            edge_attr = np.stack(edge_attr)

            # add mirrored edges
            edge_index = np.concatenate((edge_index, np.flip(edge_index, axis=0)), axis=1)
            edge_attr = np.concatenate((edge_attr, -edge_attr), axis=0)
            edge_attr[:, -1] = np.abs(edge_attr[:, -1])

            # create tensors for static variables
            node_type = torch.tensor(traj['node_type'][0], dtype=torch.int64)
            one_hot = F.one_hot(node_type.squeeze(), len(NodeType))
            edge_index = torch.tensor(edge_index, dtype=torch.int32)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

            # iterate over time steps
            for t in range(meta['trajectory_length'] - 2):
                # create tensors for dynamic variables
                velocity = torch.tensor(traj['velocity'][t], dtype=torch.float32)
                x = torch.cat([velocity, one_hot], dim=1)
                y = torch.tensor(traj['target|velocity'][t], dtype=torch.float32)
                graph = Data(x, edge_index, edge_attr, y, node_type=node_type)
                if noise:
                    add_noise(graph)
                sequence.append(graph)
            data[group] = sequence
            sequence = []
    torch.save(data, os.path.join(save_dir, split + '.pt'))


def main():
    load_dir = 'data/cylinder_flow'
    save_dir = 'data/cylinder_flow'
    # to_pyg(load_dir, save_dir, 'train', noise=True)
    # to_pyg(load_dir, save_dir, 'valid')
    # to_pyg(load_dir, save_dir, 'test')


if __name__ == '__main__':
    main()