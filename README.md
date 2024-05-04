# Solving PDEs on irregular meshes using GNNs

The code in this repository explores an alternative approach to incorporating PDE residuals for unsupervised learning with GNNs. It considers a transfer-learning framework broken into two stages:

1. Using supervised learning, pretrain a GNN to approximate the gradient of a scalar field represented on an irregular mesh.
2. With the weights of the pretrained GNN fixed, use forward passes to compute the terms of a PDE residual, and minimize the residual with respect to the input fields to obtain the solution.

The `data.py` file generates training data for the pretraining stage. It can be configured on the command line, e.g.

```bash
./data.py --type irregular --train 1000 --val 100 --double
```

generates a training set of 1000 samples and 100 validation samples using double precision. The `--type` flag specifies the kind of meshes to generate, where the options are `line` for a one-dimensional line mesh, `block` for a graded block mesh, and `irregular` for a triangular mesh.

The `train.py` file can be used to train one of our models, which are contained in the files `two.py`, `total.py`, `simple.py`, `gnn.py`, and  `medium.py`. The latter is the one used in our project. An example training configuration is 

```bash
 ./train.py --model medium --epochs 100 --lr 1e-2 --gamma 0.99
```

which trains the `medium.py` model for 100 epochs, using an initial learning rate of $10^{-2}$ and an exponential learning rate decay of $\gamma = 0.99$.

The results of training can be viewed with `view.py`. Calling

```bash
./view.py --index 0
```

can be used to cycle through different samples, specified by the `--index` flag, of the test set.

Once a model is trained, the second part of the transfer-learning task can be tested by running `solve.py` in the `solve` folder.

As a byproduct of this project, we rebuilt the MeshGraphs architecture from [[1]](#1) using PyTorch Geometric. The code for this implementation is contained in the `mgn` folder.

## References

<a id="1">[1]</a>: Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W. (2021). Learning Mesh-Based Simulation with Graph Networks (arXiv:2010.03409). arXiv. http://arxiv.org/abs/2010.03409


