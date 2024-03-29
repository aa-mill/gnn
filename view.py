import torch
import numpy as np
import matplotlib.pyplot as plt
import regular
import irregular
import pickle
import simple


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def testLine(model, data, index=0):
    line = data[index]
    model.eval()
    with torch.no_grad():
        data = line.line2Graph(nps=2).to(device)
        x, _, _ = model(data.x, data.edge_index, data.edge_attr)
        x[0][data.mask] = data.bcs
        field = x[0].detach().cpu().numpy()[:, 0]
        weights = weightArray(data, x[1].detach().cpu().numpy())
        print(weights[:10, :])
    line.plotLine({'data': field, 
                   'ls': '-', 
                   'label': 'GNN'}, 'figs/1d.png')


def testBlock(model, data, index=0):
    mesh = data[index]
    model.eval()
    with torch.no_grad():
        data = mesh.mesh2Graph().to(device)
        x, _, _ = model(data.x, data.edge_index, data.edge_attr)
        x[data.mask] = data.bcs
        field = x.detach().cpu().numpy()[:, 0]
        field = field.reshape(mesh.F.shape)
    vmin = min(mesh.Fx.min(), field.min())
    vmax = max(mesh.Fx.max(), field.max())
    regular.plotMesh(mesh.Fx, 'figs/truth.png', vmin, vmax)
    regular.plotMesh(field, 'figs/pred.png', vmin, vmax)


def testIrregular(model, data, index=0):
    mesh = data[index]
    model.eval()
    with torch.no_grad():
        data = irregular.mesh2Graph(mesh).to(device)
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


def lineConvergence(model):
    errors = []
    h = [2**-i for i in np.linspace(4, 12, 15)]
    for i in h:
        line = regular.Line(x0=0, n=int(1/i), g=1, l=1)
        model.eval()
        with torch.no_grad():
            data = line.line2Graph(nps=2).to(device)
            x, _, _ = model(data.x, data.edge_index, data.edge_attr)
            x[0][data.mask] = data.bcs
            field = x[0].detach().cpu().numpy()[:, 0]
        error = np.linalg.norm(field - line.Fx, np.inf)
        fd = np.linalg.norm(np.gradient(line.F, line.X) - line.Fx, np.inf)
        errors.append((error, fd))
    errors = np.array(errors)
    fig, ax = plt.subplots()
    ax.loglog(h, errors[:, 0], '.-', label='GNN')
    ax.loglog(h, errors[:, 1], 'k.-', label='FD')
    ax.loglog(h, [i**2 for i in h], 'k--', label=r'$h^2$')
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$\|\cdot\|_\infty$')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figs/convergence.png', dpi=250)


def main():
    with open('data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    model = getattr(globals()['simple'], 'model').to(device)
    model.load_state_dict(torch.load('trained_model.pt'))
    testLine(model, test_data, 2)


if __name__ == '__main__':
    main()