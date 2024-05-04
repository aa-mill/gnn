#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import regular
import irregular
import pickle
import simple, medium, two, gnn
import argparse
from findiff import FinDiff
from total import EncodeProcessDecode


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def testLine(model, data, nps, index=0):
    line = data[index]
    model.eval()
    with torch.no_grad():
        data = line.line2Graph(nps=nps, double=True).to(device)
        x, _, _ = model(data.x, data.edge_index, data.edge_attr)
        x[0][data.mask] = data.bcs
        field = x[0].detach().cpu().numpy()[:, 0]
        weights = weightArray(data, x[1].detach().cpu().numpy())
        print(weights[:10, :])
    line.plotLine({'data': field, 
                   'ls': '-', 
                   'color': '#03A9F4',
                   'label': 'GNN'}, 'figs/1d.png')


def testBlock(model, data, double, index=0):
    mesh = data[index]
    model.eval()
    with torch.no_grad():
        data = mesh.mesh2Graph(double).to(device)
        x, _, _ = model(data.x, data.edge_index, data.edge_attr)
        x[data.mask] = data.bcs
        field = x.detach().cpu().numpy()[:, 0]
        field = field.reshape(mesh.F.shape)
    vmin = min(mesh.Fx.min(), field.min())
    vmax = max(mesh.Fx.max(), field.max())
    mesh.plotMesh(mesh.Fx, 'figs/truth.png', vmin, vmax)
    mesh.plotMesh(field, 'figs/pred.png', vmin, vmax)


def testIrregular(model, data, double, index=0):
    mesh = data[index]
    model.eval()
    with torch.no_grad():
        data = irregular.mesh2Graph(mesh, double).to(device)
        x, _, _ = model(data.x, data.edge_index, data.edge_attr)
        x[data.mask] = data.bcs
        field = x.detach().cpu().numpy()[:, 0]
    Fx = mesh.point_data['Fx']
    vmin = min(Fx.min(), field.min())
    vmax = max(Fx.max(), field.max())
    mesh.point_data['pred'] = field
    irregular.plotMesh(mesh, 'Fx', 'figs/truth.png', vmin, vmax)
    irregular.plotMesh(mesh, 'pred', 'figs/pred.png', vmin, vmax)


def weightArray(graph, weights):
    result = np.full((graph.num_nodes, 4), np.nan)
    for feature, (src, dest) in zip(weights, graph.edge_index.T):
        if src - dest == -2:
            col = 0
        elif src - dest == -1:
            col = 1
        elif src - dest == 1:
            col = 2
        elif src - dest == 2:
            col = 3
        result[dest, col] = feature
    return result


def lineConvergence(model, nps, double):
    errors = []
    h = [2**-i for i in np.linspace(4, 11, 15)]
    for i in h:
        line = regular.Line(x0=-np.pi, n=int(1/i), g=1, l=2*np.pi)
        model.eval()
        with torch.no_grad():
            data = line.line2Graph(nps, double).to(device)
            x, _, _ = model(data.x, data.edge_index, data.edge_attr)
            x[0][data.mask] = data.bcs
            field = x[0].detach().cpu().numpy()[:, 0]
        error = np.linalg.norm(field - line.Fx, np.inf)
        dx = line.X[1] - line.X[0]
        ddx = FinDiff(0, dx, acc=2)(line.F)
        fd = np.linalg.norm(ddx - line.Fx, np.inf)
        errors.append((error, fd))
    errors = np.array(errors)
    fig, ax = plt.subplots()
    ax.loglog(h, errors[:, 0], '.-', color='#03A9F4', label='GNN')
    ax.loglog(h, errors[:, 1], 'k.-', label='FD')
    ax.loglog(h, [i**2 for i in h], 'k--', label=r'$h^2$')
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$\|\cdot\|_\infty$')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figs/convergence.png', dpi=250)


def parse_args():
    """
    Parses command line arguments for generating graded block mesh.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Visualize performance on test sample.')
    parser.add_argument('-i', '--index', type=int, default=0,
                        help='Index of training set sample to view.')
    return parser.parse_args()


def main():
    args = parse_args()
    with open('data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    # model = getattr(globals()['simple'], 'model').to(device)
    # model.load_state_dict(torch.load('trained_model.pt'))
    # testLine(model, test_data, nps=2, index=args.index)
    # lineConvergence(model, nps=2, double=True)

    model = getattr(globals()['medium'], 'model').to(device)
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
    # ).to(device)
    model.load_state_dict(torch.load('trained_model.pt'))
    testIrregular(model, test_data, double=True, index=args.index)
    # testBlock(model, test_data, double=True, index=args.index)


if __name__ == '__main__':
    main()