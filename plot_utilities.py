import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_losses(losses, title=True):
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(len(losses)), losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(linewidth=0.2)
    if title:
        ax.set_title('Training convergence')
    return fig, ax

def plot_surface_contour(X, u, title=True):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    ax1.plot_trisurf(X[:, 0], X[:, 1], u[:, 0], 
                     cmap=plt.cm.viridis, 
                     linewidth=0.2)
    im = ax2.tricontourf(X[:, 0], X[:, 1], u[:, 0], 
                         cmap=plt.cm.viridis)
    for ax in [ax1, ax2]:
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
    ax2.set_aspect('equal')
    fig.colorbar(im, shrink=0.7)
    if title:
        fig.suptitle('Trial solution $v(x,y)$')
    return fig, [ax1, ax2]

def plot_surface(X, u):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    im = ax.plot_trisurf(X[:, 0], X[:, 1], u[:, 0], 
                        cmap=plt.cm.viridis, 
                        linewidth=0.,
                        antialiased=False,
                        edgecolor='none')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"]['linewidth'] = 0.2
    fig.colorbar(im, shrink=0.7)

    return fig, ax

def plot_contour(X, u, **plot_kwargs):
    fig, ax = plt.subplots()
    im = ax.tricontourf(X[:, 0], X[:, 1], u[:, 0], 
                        cmap=plt.cm.viridis, 
                        **plot_kwargs)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.colorbar(im)
    fig.tight_layout()
    return fig, ax

def plot_error(X, u, u_analytical, title=True, **plot_kwargs):
    error = np.abs(u[:, 0] - u_analytical(X[:, 0], X[:, 1]))
    error += 1e-16  # avoid log(0)
    fig, ax = plt.subplots()
    im = ax.tricontourf(X[:, 0], X[:, 1], error, 
                        cmap=plt.cm.viridis, 
                        norm=LogNorm(),
                        **plot_kwargs)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if title:
        ax.set_title('Absolute error $|(v - u)|$')
    fig.colorbar(im)
    fig.tight_layout()
    return fig, ax
