import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/aaronmiller/research/gnn')
import regular, medium
import torch
import torch.nn as nn
from torch_geometric.data import Data
from total import EncodeProcessDecode

# load foam data
df = pd.read_csv('ldc.csv')
df = df.rename(columns={'Points:0': 'x', 'Points:1': 'y', 
                        'U:0': 'u', 'U:1': 'v',
                        'grad(U):0': 'ux', 'grad(U):1': 'uy'})
df = df[['x', 'y', 'u', 'v', 'p', 'ux']].drop_duplicates(subset=['x', 'y'])
df = df.sort_values(by=['y', 'x'])
X, Y = np.meshgrid(df['x'].unique(), df['y'].unique())
Z = df['u'].values.reshape(X.shape)
target = df[['u', 'v', 'p']].values

def mesh2Graph(X, Y, target, double=False):
    dtype = torch.float64 if double else torch.float32
    nx, ny = X.shape[1] - 1, X.shape[0] - 1  # number of cells
    Nv = len(target)  # number of nodes
    Ne = (nx + 1)*ny + (ny + 1)*nx  # number of edges

    # these next few lines create a mask to exclude boundary nodes
    idxs = np.arange(Nv).reshape(ny + 1, nx + 1) # array of node indices
    boundaries = np.concatenate((idxs[0, :], idxs[-1, :], idxs[:, 0], idxs[:, -1])) # list of boundary indices
    mask = np.isin(idxs.flatten(), np.unique(boundaries)) # 1 for boundary nodes, 0 for interior nodes

    y = torch.tensor(target, dtype=dtype) # target values
    bcs = y[mask] # boundary conditions

    edge_index = np.zeros((2, Ne), dtype=int) # edge indices, shape (2, Ne)
    edge_attr = np.zeros((Ne, 2)) # edge features, shape (Ne, ne)
    edge_idx = 0

    # for each row
    for i in range(ny):
        # for each column
        for j in range(nx):
            # identify absolute, i.e. flattened, node index
            node_idx = j + i*(nx + 1)
            # add horizontal edge
            edge_index[:, edge_idx] = [node_idx, node_idx + 1] # add indices
            edge_attr[edge_idx, :] = [X[i, j] - X[i, j + 1], 0] # add features
            edge_idx += 1
            # add vertical edge
            edge_index[:, edge_idx] = [node_idx, node_idx + nx + 1] # indices
            Yf = np.flip(Y) # flip to match physical indexing
            edge_attr[edge_idx] = [0, Yf[i, j] - Yf[i + 1, j]] # features
            edge_idx += 1

    # add right boundary
    for i in range(ny):
        node_idx = nx + i*(nx + 1)
        edge_index[:, edge_idx] = [node_idx, node_idx + nx + 1]
        edge_attr[edge_idx, :] = [0, Yf[i, nx] - Yf[i + 1, nx]]
        edge_idx += 1

    # add bottom boundary
    for j in range(nx):
        node_idx = j + ny*(nx + 1)
        edge_index[:, edge_idx] = [node_idx, node_idx + 1]
        edge_attr[edge_idx, :] = [X[ny, j] - X[ny, j + 1], 0]
        edge_idx += 1

    # add mirrored edges
    edge_index = np.concatenate((edge_index, np.flip(edge_index, axis=0)), axis=1)
    edge_attr = np.concatenate((edge_attr, -edge_attr), axis=0)

    # compute node degrees
    degrees = np.zeros(Nv, dtype=int)
    np.add.at(degrees, edge_index[1, :], 1)

    # create PyTorch tensors
    x = torch.tensor(degrees, dtype=dtype)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=dtype)
    mask = torch.tensor(mask, dtype=torch.bool)
    # return graph in PyG format
    return Data(x, edge_index, edge_attr, y, mask=mask, bcs=bcs)

graph = mesh2Graph(X, Y, target, double=True)

# load model
model = medium.model
# model = EncodeProcessDecode(
#     node_size=2, 
#     edge_size=3, 
#     hidden_size=32, 
#     hidden_layers=2, 
#     latent_size=32, 
#     output_size=2,
#     graph_layers=5,
#     batch_size=64,
#     normalize=False
# )
model.load_state_dict(torch.load('../trained_model.pt'))

class Solver(nn.Module):
    def __init__(self, model, graph, nu):
        super(Solver, self).__init__()
        self.model = model
        self.degrees = graph.x
        self.edge_index = graph.edge_index
        self.edge_attr = graph.edge_attr
        self.mask = graph.mask
        self.bcs = graph.bcs
        self.nu = nu

    def forward(self, fields):
        u, v, p = fields.T

        # compute gradients
        grad_u, _, _ = self.model(torch.stack([u, self.degrees], dim=1), 
                                  self.edge_index, self.edge_attr)
        grad_v, _, _ = self.model(torch.stack([v, self.degrees], dim=1),
                                  self.edge_index, self.edge_attr)
        grad_p, _, _ = self.model(torch.stack([p, self.degrees], dim=1), 
                                  self.edge_index, self.edge_attr)
        
        # compute Laplacians
        grad_ux, _, _ = self.model(torch.stack([grad_u[:, 0], self.degrees], dim=1), 
                                   self.edge_index, self.edge_attr)
        grad_uy, _, _ = self.model(torch.stack([grad_u[:, 1], self.degrees], dim=1), 
                                   self.edge_index, self.edge_attr)
        grad_vx, _, _ = self.model(torch.stack([grad_v[:, 0], self.degrees], dim=1),
                                   self.edge_index, self.edge_attr)
        grad_vy, _, _ = self.model(torch.stack([grad_v[:, 1], self.degrees], dim=1), 
                                   self.edge_index, self.edge_attr)
        lap_u = grad_ux[:, 0] + grad_uy[:, 1]
        lap_v = grad_vx[:, 0] + grad_vy[:, 1]

        # compute residuals
        continuity = [grad_u[:, 0] + grad_v[:, 1]]
        momentum = [u*grad_u[:, 0] + v*grad_u[:, 1] + grad_p[:, 0] - self.nu*lap_u,
                    u*grad_v[:, 0] + v*grad_v[:, 1] + grad_p[:, 1] - self.nu*lap_v]
        residual = continuity + momentum

        return torch.stack(residual, dim=1)

# solve
solver = Solver(model, graph, nu=0.01)
fields = np.zeros_like(graph.y)
fields[graph.mask, :2] = graph.bcs[:, :2]
fields = torch.tensor(fields, requires_grad=True, dtype=torch.float64)
optimizer = torch.optim.Adam([fields], lr=1e-3)
for i in range(250):
    optimizer.zero_grad()
    residual = solver(fields)
    loss = torch.sum(residual[~graph.mask]**2)
    loss.backward()
    fields.grad[graph.mask] = 0
    optimizer.step()
    print(f'{i + 1}: {loss.item()}')

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots()
ax.plot(X, Y, 'k', lw=.5)
ax.plot(X.T, Y.T, 'k', lw=.5)
u = fields[:, 0].detach().numpy().reshape(X.shape)
grad_u, _, _ = model(torch.stack([graph.y[:, 0], graph.x], dim=1), 
                     graph.edge_index, graph.edge_attr)
grad_u = grad_u.detach().numpy()[:, 0].reshape(X.shape)
exact = df['u'].values.reshape(X.shape)
vmin = min(u.min(), exact.min())
vmax = max(u.max(), exact.max())
countour = ax.contourf(X, Y, u, 100, cmap='coolwarm', vmin=vmin, vmax=vmax)
cb = fig.colorbar(countour, ax=ax)
cb.set_label(r'$u_x$' + ' [m/s]')
ax.set_aspect('equal')
ax.set_xlabel(r'$x$' + ' [m]')
ax.set_ylabel(r'$y$' + ' [m]')
ax.set_title('Prediction')
fig.tight_layout()
fig.savefig('solution.png', dpi=250)