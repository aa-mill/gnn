import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/aaronmiller/research/gnn')
import regular, medium
import torch
import torch.nn as nn
from torch_geometric.data import Data

# read experimental data
U = pd.read_csv('u.txt', delim_whitespace=True)
V = pd.read_csv('v.txt', delim_whitespace=True)

# load foam data
df = pd.read_csv('ldc.csv')
df = df.rename(columns={'Points:0': 'x', 'Points:1': 'y', 'U:0': 'u'})
df = df[['x', 'y', 'u']].drop_duplicates(subset=['x', 'y'])
cl = df[df['x'] == 0.5][['y', 'u']].sort_values(by=['y'])

# plot data
fig, ax = plt.subplots()
ax.scatter(U['y'], U['100'], color='k', label='Ghia, Ghia, et al.')
ax.plot(cl['y'], cl['u'], color='#03A9F4', label='OpenFOAM')
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$u$')
ax.set_title('Re = 100')
ax.grid('on')
ax.legend()
fig.tight_layout()
fig.savefig('validate.png', dpi=250)