import pickle
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import torch
from torch_geometric.data import Data
from collections import Counter
from tqdm import tqdm


def buildMesh():
    """
    Creates a random mesh using pygmsh.

    Returns:
        mesh (pyg.Data): A meshio Mesh object.
    """
    rng = np.random.default_rng()
    with pygmsh.geo.Geometry() as geom:
        bl = list((rng.uniform(-0.1, 0, 2)))
        br = list((rng.uniform(0, 1), rng.uniform(-1, 0)))
        tl = list((rng.uniform(-1, 0), rng.uniform(0, 1)))
        tr = list((rng.uniform(0, 1, 2)))
        geom.add_polygon(
            [bl, br, tr, tl],
            mesh_size=0.025,
        )
        return geom.generate_mesh()


def plotMesh(mesh, field, output_path=None, vmin=None, vmax=None):
    """
    Plots mesh and scalar field.

    Parameters:
        mesh (pyg.Data): A meshio Mesh object.
        field (str): Field of the Mesh object to plot.
        output_path (str): Path to save the plot.
        vmin (float): Minimum value of the colorbar.
        vmax (float): Maximum value of the colorbar.

    Returns:
        vmin (float): Minimum value of the colorbar.
        vmax (float): Maximum value of the colorbar.
    """
    fig, ax = plt.subplots()
    for cell in mesh.cells:
        if cell.type == "triangle":
            x, y = mesh.points[:, 0], mesh.points[:, 1]
            print(vmin, vmax)
            contour = ax.tricontourf(x, y, cell.data, mesh.point_data[field], 
                                     100, cmap='plasma', vmin=vmin, vmax=vmax)
            ax.triplot(x, y, cell.data, color='k', lw=0.75)
            cb = fig.colorbar(contour, ax=ax)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect('equal')
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=250)
    else:
        fig.savefig('irregular.png', dpi=250)
    return cb.vmin, cb.vmax


def addFields(mesh):
    """
    Generates random field F and its gradient (Fx, Fy) over mesh.

    Returns:
        F (ndarray): Field evaluated at (self.X, self.Y).
        Fx (ndarray): Gradient w.r.t x.
        Fy (ndarray): Gradient w.r.t y.
    """
    # generate random functions
    x, y = sp.symbols('x y')
    # basis = [1, sp.sin(x), sp.sin(y), sp.sin(x)*sp.sin(y), x, y, x*y]
    basis = [x, y, sp.sin(x), sp.sin(y), sp.sin(x)*sp.sin(y), x*y]
    coeffs = np.random.uniform(-1, 1, len(basis))
    func = sum([coeff*b for coeff, b in zip(coeffs, basis)])

    # get analytical functions and gradients
    f = sp.lambdify((x, y), func, 'numpy')
    fx = sp.lambdify((x, y), sp.diff(func, x), 'numpy')
    fy = sp.lambdify((x, y), sp.diff(func, y), 'numpy')

    # compute function and gradients over mesh
    X, Y = mesh.points[:, 0], mesh.points[:, 1]
    F = f(X, Y)
    Fx = fx(X, Y)
    Fy = fy(X, Y)

    # add fields to Mesh object
    mesh.point_data['F'] = F
    mesh.point_data['Fx'] = Fx
    mesh.point_data['Fy'] = Fy


def mesh2Graph(mesh):
    """
    Converts meshio Mesh object to a graph in PyG format.

    Args:
        mesh (Mesh): The input mesh object.

    Returns:
        torch_geometric.data.Data: A PyG Data object representing the mesh.
    """
    node_attr = mesh.point_data['F'].reshape(-1, 1)

    # first get edges for each triangle
    # contains many duplicates but makes it easy to identify boundary edges
    all_edges = []
    for cell in mesh.cells: # each cell has a type and data
        if cell.type == 'triangle': 
            for tri in cell.data: # each triangle has 3 points
                # get triangle edges
                # min/max imposes ordering to facilitate duplicate removal
                edges = [(min(tri[i], tri[(i + 1) % 3]), 
                          max(tri[i], tri[(i + 1) % 3])) 
                          for i in range(3)]
                all_edges.extend(edges)
    # place unique edges in index array
    edge_index = np.array(list(set(all_edges)), dtype=int).T

    # edges that only appear once must be boundary edges
    edge_count = Counter(all_edges)
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    boundary_nodes = set(node for edge in boundary_edges for node in edge)
    # mask for boundary nodes, 1 if boundary, 0 otherwise
    mask = np.isin(np.arange(len(mesh.points)), list(boundary_nodes))
    y = np.stack((mesh.point_data['Fx'], mesh.point_data['Fy']), axis=1)
    bcs = y[mask]

    # edge attributes are relative x and y displacements
    edge_attr = np.zeros((len(edge_index.T), 2))
    for idx, (src, dest) in enumerate(edge_index.T):
        dx = mesh.points[src][0] - mesh.points[dest][0]
        dy = mesh.points[src][1] - mesh.points[dest][1]
        edge_attr[idx, :] = [dx, dy]

    # compute node degrees
    degrees = np.zeros(len(mesh.points), dtype=int)
    np.add.at(degrees, edge_index[1, :], 1)
    node_attr = np.concatenate((node_attr, degrees.reshape(-1, 1)), axis=1)

    # add mirrored edges
    edge_index = np.concatenate((edge_index, np.flip(edge_index, axis=0)), axis=1)
    edge_attr = np.concatenate((edge_attr, -edge_attr), axis=0)

    # create PyTorch tensors
    x = torch.tensor(node_attr, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)
    bcs = torch.tensor(bcs, dtype=torch.float32)
    # return graph in PyG format
    return Data(x, edge_index, edge_attr, y, mask=mask, bcs=bcs)
    

def createData(num_samples, output_path=None, raw=False):
    """
    Creates a dataset of size num_samples, where each sample is a Data object.

    Parameters:
        num_samples (int): Number of samples to generate.
        raw (bool): If true, returns raw mesh objects, otherwise returns PyG Data objects.

    Returns:
        data (list): A list of Data objects, each representing a mesh.
    """
    rng = np.random.default_rng()
    data = []
    print('Generating data...')
    for _ in tqdm(range(num_samples)):
        mesh = buildMesh()
        addFields(mesh)
        if raw:
            data.append(mesh)
        else:
            data.append(mesh2Graph(mesh))
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open('data.pkl', 'wb') as f:
            pickle.dump(data, f)

            
if __name__ == '__main__':
    mesh = buildMesh()
    addFields(mesh)
    for cell in mesh.cells:
        if cell.type == 'triangle':
            print(cell.data)
    mesh2Graph(mesh)
    plotMesh(mesh, field='F')