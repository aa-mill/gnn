import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import argparse
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pickle


class Mesh():
    """
    Object for generating and visualizing graded block meshes.

    Args:
        x0 (float): Starting x coordinate.
        y0 (float): Starting y coordinate.
        nx (int): Number of cells in x direction.
        ny (int): Number of cells in y direction.
        gx (float): Grading in x direction.
        gy (float): Grading in y direction.
        lx (float): Length in x direction.
        ly (float): Length in y direction.

    Attributes:
        x0 (float): Starting x coordinate.
        y0 (float): Starting y coordinate.
        nx (int): Number of cells in x direction.
        ny (int): Number of cells in y direction.
        gx (float): Grading in x direction.
        gy (float): Grading in y direction.
        lx (float): Length in x direction.
        ly (float): Length in y direction.
        X (ndarray): Mesh grid for x values.
        Y (ndarray): Mesh grid for y values.
        F (ndarray): Field evaluated at (X, Y).
        Fx (ndarray): Gradient of the field with respect to x.
        Fy (ndarray): Gradient of the field with respect to y.

    Methods:
        __init__(): Initializes the Mesh object.
        __str__(): Outputs description of the mesh.
        gradedValues(): Generates an array of graded values.
        buildMesh(): Builds a mesh grid using graded values of x and y.
        plotMesh(): Plots the mesh defined by the X and Y grids.
        mesh2Graph(): Converts the mesh to a graph representation.
        getFields(): Generates random fields F and its gradient (Fx, Fy) over mesh.
    """
    def __init__(self, x0, y0, nx, ny, gx, gy, lx, ly):
        self.x0 = x0  # starting x coordinate
        self.y0 = y0  # starting y coordinate
        self.nx = nx  # number of cells in x direction
        self.ny = ny  # number of cells in y direction
        self.gx = gx  # grading in x direction
        self.gy = gy  # grading in y direction
        self.lx = lx  # length in x direction
        self.ly = ly  # length in y direction
        self.X, self.Y = self.buildMesh() 
        self.F, self.Fx, self.Fy = self.getFields()

    def __str__(self):
        """
        Outputs description of the mesh.

        Returns:
            str: String representation of the mesh.
        """
        return ('----- Mesh Description -----\n'
                f'Domain: [{self.x0:.3f}, {self.x0 + self.lx:.3f}] '
                f'x [{self.y0:.3f}, {self.y0 + self.ly:.3f}]\n'
                f'Cells: {self.nx*self.ny}\n'
                f'Nodes: {(self.nx + 1)*(self.ny + 1)}\n'
                f'Edges: {(self.nx + 1)*self.ny + (self.ny + 1)*self.nx}\n'
                '----------------------------')

    def gradedValues(self, start, length, grade, cells):
        """
        Generates an array of graded values.

        Args:
            start (float): Lower boundary of domain.
            length (float): Length of domain.
            grade (float): Ratio of last cell length to first cell length.
            cells (int): Number of cells along dimension.

        Returns:
            ndarray: Array of graded values.
        """
        r = grade**(1/(cells - 1))
        dx = length*(1 - r)/(1 - r**cells)
        return np.array([start + dx*(1 - r**i)/(1 - r) for i in range(cells + 1)])
    
    def buildMesh(self):
        """
        Builds a mesh grid using graded values of x and y.

        Returns:
            X (ndarray): Mesh grid for x values.
            Y (ndarray): Mesh grid for y values.
        """
        x = self.gradedValues(self.x0, self.lx, self.gx, self.nx)
        y = self.gradedValues(self.y0, self.ly, self.gy, self.ny)
        X, Y = np.meshgrid(x, y)
        return X, Y
    
    def plotMesh(self, field=None, output_path=None):
        """
        Plots the mesh defined by the X and Y grids. Optionally, plots the fields.

        Returns:
            None
        """
        X, Y = self.X, self.Y
        fig, ax = plt.subplots()
        ax.plot(X, Y, 'k')
        ax.plot(X.T, Y.T, 'k')
        if field is not None:
            contour = plt.contourf(X, Y, field, 100, cmap='plasma')
            fig.colorbar(contour, ax=ax)
        ax.set_xticks([X[0, 0], X[0, -1]])
        ax.set_yticks([Y[0, 0], Y[-1, 0]])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('equal')
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=250)
        else:
            fig.savefig('mesh.png', dpi=250)

    def mesh2Graph(self):
        """
        Converts the mesh to a graph representation.

        Returns:
            node_attr (ndarray): Node attributes, shape (Nv, nv), where Nv = number of nodes
                and nv = number of attributes per node.
            edge_index (ndarray): Edge indices, shape (2, Ne), where Ne = number of edges
                and ne = number of attributes per edge.
            edge_attr (ndarray): Edge attributes, shape (Ne, ne).
        """
        Nv = self.F.size  # number of nodes
        Ne = (self.nx + 1)*self.ny + (self.ny + 1)*self.nx  # number of edges

        node_attr = self.F.reshape(Nv, 1) # node features, shape (Nv, nv)
        y = np.concatenate((self.Fx.reshape(Nv, 1), self.Fy.reshape(Nv, 1)), axis=1)

        edge_index = np.zeros((2, Ne), dtype=int) # edge indices, shape (2, Ne)
        edge_attr = np.zeros((Ne, 2)) # edge features, shape (Ne, ne)
        edge_idx = 0

        # for each row
        for i in range(self.ny):
            # for each column
            for j in range(self.nx):
                # identify absolute, i.e. flattened, node index
                node_idx = j + i*(self.nx + 1)
                # add horizontal edge
                edge_index[:, edge_idx] = [node_idx, node_idx + 1] # add indices
                edge_attr[edge_idx, :] = [self.X[i, j] - self.X[i, j + 1], 0] # add features
                edge_idx += 1
                # add vertical edge
                edge_index[:, edge_idx] = [node_idx, node_idx + self.nx + 1] # indices
                Y = np.flip(self.Y) # flip to match physical indexing
                edge_attr[edge_idx] = [0, Y[i, j] - Y[i + 1, j]] # features
                edge_idx += 1

        # add right boundary
        for i in range(self.ny):
            node_idx = self.nx + i*(self.nx + 1)
            edge_index[:, edge_idx] = [node_idx, node_idx + self.nx + 1]
            Y = np.flip(self.Y) # flip to match physical indexing
            edge_attr[edge_idx, :] = [0, Y[i, self.nx] - Y[i + 1, self.nx]]
            edge_idx += 1

        # add bottom boundary
        for j in range(self.nx):
            node_idx = j + self.ny*(self.nx + 1)
            edge_index[:, edge_idx] = [node_idx, node_idx + 1]
            edge_attr[edge_idx, :] = [self.X[self.ny, j] - self.X[self.ny, j + 1], 0]
            edge_idx += 1

        # add mirrored edges
        edge_index = np.concatenate((edge_index, np.flip(edge_index, axis=0)), axis=1)
        edge_attr = np.concatenate((edge_attr, -edge_attr), axis=0)

        # create PyTorch tensors
        x = torch.tensor(node_attr, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # return graph in PyG format
        return Data(x, edge_index, edge_attr, y)

    def getFields(self):
        """
        Generates random fields F and its gradient (Fx, Fy) over mesh.

        Returns:
            F (ndarray): Field evaluated at (self.X, self.Y).
            Fx (ndarray): Gradient w.r.t x.
            Fy (ndarray): Gradient w.r.t y.
        """
        # generate random functions
        x, y = sp.symbols('x y')
        basis = [1, sp.sin(x), sp.sin(y), sp.sin(x)*sp.sin(y), x, y, x*y]
        coeffs = np.random.uniform(-1, 1, len(basis))
        func = sum([coeff*b for coeff, b in zip(coeffs, basis)])

        # get analytical functions and gradients
        f = sp.lambdify((x, y), func, 'numpy')
        fx = sp.lambdify((x, y), sp.diff(func, x), 'numpy')
        fy = sp.lambdify((x, y), sp.diff(func, y), 'numpy')

        # compute function and gradients over mesh
        F = f(self.X, self.Y)
        Fx = fx(self.X, self.Y)
        Fy = fy(self.X, self.Y)

        return F, Fx, Fy


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
        x0, y0 = rng.uniform(-10, 10, size=2)
        nx, ny = rng.integers(10, 100, size=2)
        gx, gy = rng.uniform(.1, 10, size=2)
        lx, ly = rng.uniform(1, 10, size=2)
        mesh = Mesh(x0, y0, nx, ny, gx, gy, lx, ly)
        if raw:
            data.append(mesh)
        else:
            data.append(mesh.mesh2Graph())
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open('data.pkl', 'wb') as f:
            pickle.dump(data, f)
    

def parse_args():
    """
    Parses command line arguments for generating graded block mesh.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Generate graded block mesh')
    parser.add_argument('--x0', type=int, default=0, 
                        help='Starting x coordinate')
    parser.add_argument('--y0', type=int, default=0, 
                        help='Starting y coordinate')
    parser.add_argument('--nx', type=int, default=3, 
                        help='Number of horizontal cells')
    parser.add_argument('--ny', type=int, default=3, 
                        help='Number of vertical cells')
    parser.add_argument('--gx', type=int, default=.1, 
                        help='Grading in x direction')
    parser.add_argument('--gy', type=int, default=5, 
                        help='Grading in y direction')
    parser.add_argument('--lx', type=int, default=3, 
                        help='Length in x direction')
    parser.add_argument('--ly', type=int, default=2, 
                        help='Length in y direction')
    return parser.parse_args()


def main():
    args = parse_args()
    mesh = Mesh(**vars(args))
    mesh.plotMesh(field=mesh.F)
    print(mesh)
    graph = mesh.mesh2Graph()
    print(graph.x.numpy())
    print(np.hstack((graph.edge_index.numpy().T, graph.edge_attr.numpy())))
    print(mesh.X)
    print(np.flip(mesh.Y))


if __name__ == '__main__':
    main()