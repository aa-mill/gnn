#!/usr/bin/env python3
import torch
from model import EncodeProcessDecode
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py


# set device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def plotSlice(sample_index, rollout, t):
    # get mesh data and target
    with h5py.File('data/cylinder_flow/test.h5', 'r') as file:
        mesh = file[str(sample_index)]
        cells = mesh['cells'][0]
        x, y = mesh['mesh_pos'][0].T
        truth = mesh['target|velocity'][t, :, 0]

    # create figure
    fig, axes = plt.subplots(2, 1)
    vel_x, vel_y = rollout[t][:, 0], rollout[t][:, 1]
    countour = axes[0].tricontourf(x, y, cells, truth, 100, cmap='plasma')
    cb = fig.colorbar(countour, ax=axes, orientation='horizontal')
    axes[1].tricontourf(x, y, cells, vel_x, 100, cmap='plasma', vmin=cb.vmin, vmax=cb.vmax)
    for ax in axes:
        ax.triplot(x, y, cells, color='k', lw=0.75)
        ax.set_aspect('equal')
    fig.savefig('visual.png', dpi=250)

    return


def plotVideo(sample_index, rollout, max_time=10):
    # get mesh data and target
    with h5py.File('data/cylinder_flow/valid.h5', 'r') as file:
        mesh = file[str(sample_index)]
        cells = mesh['cells'][0]
        x, y = mesh['mesh_pos'][0].T
        truth = mesh['target|velocity'][:max_time, :, 0]

    # create initial figures
    fig, axes = plt.subplots(2, 1)
    contour = axes[0].tricontourf(x, y, cells, truth[0], 100, cmap='plasma')
    cb = fig.colorbar(contour, ax=axes, location='bottom')
    cb.set_label(r'$v_x$')
    axes[1].tricontourf(x, y, cells, rollout[0][:, 0], 100, 
                        cmap='plasma', vmin=cb.vmin, vmax=cb.vmax)
    
    # format figures
    for ax in axes:
        ax.triplot(x, y, cells, color='k', lw=0.15)
        ax.set_aspect('equal')

    # update function
    def update(t):
        axes[0].tricontourf(x, y, cells, truth[t], 100, 
                            cmap='plasma', vmin=cb.vmin, vmax=cb.vmax)
        axes[1].tricontourf(x, y, cells, rollout[t][:, 0], 100,
                            cmap='plasma', vmin=cb.vmin, vmax=cb.vmax)
        for ax in axes:
            ax.triplot(x, y, cells, color='k', lw=0.15)
    
    # create animation
    print('Creating animation...')
    anim = FuncAnimation(fig, update, frames=max_time, repeat=False)
    anim.save('visual.mp4', dpi=350)
    
    return


def evaluate(model, initial_state, rollout_len):
    # set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # get initial state
        state = initial_state.clone().to(device)

        # rollout
        rollout = [state.x[:, :2].cpu().detach().numpy()]
        edge_index = state.edge_index
        edge_attr = state.edge_attr
        print('Starting eval...')
        for _ in range(rollout_len - 1):
            diff = model(state.x, edge_index, edge_attr)
            state.x[:, :2] += model.output_normalizer.inverse(diff)
            rollout.append(state.x[:, :2].cpu().detach().numpy())

    return rollout


def main():

    # create model
    model = EncodeProcessDecode(
        node_size=9, 
        edge_size=3, 
        hidden_size=128, 
        hidden_layers=2, 
        latent_size=128, 
        output_size=2,
        graph_layers=15,
        batch_size=32,
        normalize=True
    ).to(device)
    
    # load model and test set
    model.load_state_dict(torch.load('chkpts/chkpt_112500.pt'))
    test_data = torch.load('data/cylinder_flow/valid.pt')

    # evaluate model
    sample_index = 0
    initial_state = test_data[str(sample_index)][0]
    rollout_len = 598
    rollout = evaluate(model, initial_state, rollout_len)

    # plot results
    plotVideo(sample_index, rollout, max_time=rollout_len)

if __name__ == '__main__':
    main()