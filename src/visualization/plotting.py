"""
Plotting utilities for photoacoustic computed tomography visualization.

This module provides functions for creating publication-quality plots and animations
of reconstruction results, including comparison plots, convergence animations, and
custom colormaps for highlighting reconstruction quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional, Union


def create_colormap(data: np.ndarray, gt: np.ndarray) -> Tuple[LinearSegmentedColormap, mcolors.Normalize]:
    """
    Create a custom colormap for comparing reconstruction data with ground truth.
    
    This function creates a colormap that highlights differences between reconstructed
    data and ground truth by using different color schemes for values outside the
    ground truth range.

    Parameters
    ----------
    data : np.ndarray
        Reconstructed data array
    gt : np.ndarray
        Ground truth data array

    Returns
    -------
    tuple
        (colormap, normalizer) - Custom colormap and normalization object

    Examples
    --------
    >>> cmap_mu_r, norm_mu_r = create_colormap(mu_r, mu)
    >>> cmap_c_r, norm_c_r = create_colormap(c_r, c)
    >>> plt.imshow(mu_r, cmap=cmap_mu_r, norm=norm_mu_r)
    """
    n_colors = 256
    vmin_data, vmax_data = np.min(data), np.max(data)
    vmin_gt, vmax_gt = np.min(gt), np.max(gt)

    vmax = max(vmax_data, vmax_gt)
    vmin = min(vmin_data, vmin_gt)
    vrange = vmax - vmin

    # Colors for ground truth range
    q = (vmax_gt - vmin_gt) / vrange 
    n_q = int(q * n_colors)
    colors_in_range = plt.cm.viridis(np.linspace(0., 1, n_q)) 
    colors = [colors_in_range]
    
    # Colors for values above ground truth range
    if vmax_data > vmax_gt:
        r = (vmax_data - vmax_gt) / vrange
        n_r = int(r * n_colors)
        colors_above = plt.cm.YlOrBr(np.linspace(0+.4, r+.4, n_r)) 
        colors.append(colors_above)

    # Colors for values below ground truth range
    if vmin_data < vmin_gt:
        p = (vmin_gt - vmin_data) / vrange
        n_p = int(p * n_colors)
        colors_below = plt.cm.BuPu(np.linspace(0+.6, p+.6, n_p))
        colors.insert(0, colors_below)

    colors = np.vstack(colors)
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def plot_recon_animation(
    mu_rs: np.ndarray, 
    c_rs: np.ndarray, 
    mu: np.ndarray, 
    c: np.ndarray, 
    save_path: str, 
    repeat_delay: int = 1000,
    fps: int = 6,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Create an animation showing reconstruction convergence over iterations.

    Parameters
    ----------
    mu_rs : np.ndarray
        Sequence of absorption coefficient reconstructions over iterations
    c_rs : np.ndarray
        Sequence of sound speed reconstructions over iterations
    mu : np.ndarray
        Ground truth absorption coefficient
    c : np.ndarray
        Ground truth sound speed
    save_path : str
        Path to save the animation file
    repeat_delay : int, optional
        Delay before repeating animation (ms). Set to 0 to disable repeat.
    fps : int, optional
        Frames per second for the animation
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, width_ratios=[0.05, 1, 1, 0.05], height_ratios=[1, 1])

    # Axes for the images
    ax1 = fig.add_subplot(gs[0, 1])  # mu_r
    ax2 = fig.add_subplot(gs[0, 2])  # c_r
    ax3 = fig.add_subplot(gs[1, 1])  # mu (GT)
    ax4 = fig.add_subplot(gs[1, 2])  # c (GT)

    # Axes for the colorbars
    cax1 = fig.add_subplot(gs[0, 0])  # mu_r colorbar
    cax2 = fig.add_subplot(gs[1, 0])  # mu (GT) colorbar
    cax3 = fig.add_subplot(gs[0, 3])  # c_r colorbar
    cax4 = fig.add_subplot(gs[1, 3])  # c (GT) colorbar

    # Create custom colormaps
    cmap_mu_r, norm_mu_r = create_colormap(mu_rs, mu)
    cmap_c_r, norm_c_r = create_colormap(c_rs, c)

    # Create animation frames
    ims = []
    num_it = len(mu_rs)
    for i in range(num_it):
        title = ax1.text(
            1.1, 1.1, f'Iteration {i}', 
            size=plt.rcParams["axes.titlesize"],
            ha="center", transform=ax1.transAxes, animated=True
        )

        im0 = ax1.imshow(mu_rs[i], cmap=cmap_mu_r, norm=norm_mu_r, animated=True)
        im1 = ax2.imshow(c_rs[i], cmap=cmap_c_r, norm=norm_c_r, animated=True)
        im2 = ax3.imshow(mu, cmap='viridis', animated=True)
        im3 = ax4.imshow(c, cmap='viridis', animated=True)
        
        ims.append([im0, im1, im2, im3, title])

    # Add repeat frames if requested
    if repeat_delay:
        for _ in range(6): 
            ims.append([im0, im1, im2, im3, title])

    # Add colorbars
    plt.colorbar(im0, cax=cax1)
    plt.colorbar(im1, cax=cax3)
    plt.colorbar(im2, cax=cax2)
    plt.colorbar(im3, cax=cax4)

    # Set titles
    ax1.set_title('μ (Reconstructed)')
    ax2.set_title('c (Reconstructed)')
    ax3.set_title('μ (Ground Truth)')
    ax4.set_title('c (Ground Truth)')

    # Remove axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axis('off')

    # Create and save animation
    ani = animation.ArtistAnimation(
        fig, ims, blit=True, 
        repeat=bool(repeat_delay), 
        repeat_delay=repeat_delay
    )
    ani.save(save_path, writer='ffmpeg', fps=fps)
    plt.show()


def plot_reconstruction_comparison(
    mu_r: np.ndarray,
    c_r: np.ndarray,
    mu_gt: np.ndarray,
    c_gt: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    titles: Optional[Tuple[str, ...]] = None
) -> plt.Figure:
    """
    Create a side-by-side comparison plot of reconstruction vs ground truth.

    Parameters
    ----------
    mu_r : np.ndarray
        Reconstructed absorption coefficient
    c_r : np.ndarray
        Reconstructed sound speed
    mu_gt : np.ndarray
        Ground truth absorption coefficient
    c_gt : np.ndarray
        Ground truth sound speed
    save_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height) in inches
    titles : tuple, optional
        Custom titles for the subplots

    Returns
    -------
    plt.Figure
        The created figure object
    """
    if titles is None:
        titles = ('μ (Reconstructed)', 'μ (Ground Truth)', 
                 'c (Reconstructed)', 'c (Ground Truth)')

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Create custom colormaps
    cmap_mu, norm_mu = create_colormap(mu_r, mu_gt)
    cmap_c, norm_c = create_colormap(c_r, c_gt)

    # Plot reconstructions and ground truth
    im1 = axes[0, 0].imshow(mu_r, cmap=cmap_mu, norm=norm_mu)
    axes[0, 0].set_title(titles[0])
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    im2 = axes[0, 1].imshow(mu_gt, cmap='viridis')
    axes[0, 1].set_title(titles[1])
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    im3 = axes[1, 0].imshow(c_r, cmap=cmap_c, norm=norm_c)
    axes[1, 0].set_title(titles[2])
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    im4 = axes[1, 1].imshow(c_gt, cmap='viridis')
    axes[1, 1].set_title(titles[3])
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_convergence_curves(
    losses: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = True
) -> plt.Figure:
    """
    Plot convergence curves for reconstruction losses.

    Parameters
    ----------
    losses : np.ndarray
        Array of loss values over iterations
    save_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height) in inches
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis

    Returns
    -------
    plt.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = np.arange(len(losses))
    ax.plot(iterations, losses, 'b-', linewidth=2, label='Reconstruction Loss')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Reconstruction Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_illumination_patterns(
    att_masks: np.ndarray,
    angles: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    max_patterns: int = 12
) -> plt.Figure:
    """
    Plot multiple illumination patterns in a grid.

    Parameters
    ----------
    att_masks : np.ndarray
        Array of attenuation masks for different illumination angles
    angles : np.ndarray
        Corresponding illumination angles
    save_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height) in inches
    max_patterns : int, optional
        Maximum number of patterns to display

    Returns
    -------
    plt.Figure
        The created figure object
    """
    n_patterns = min(len(att_masks), max_patterns)
    n_cols = 4
    n_rows = (n_patterns + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_patterns):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        im = ax.imshow(att_masks[i].squeeze(), cmap='viridis')
        ax.set_title(f'Angle: {angles[i]:.1f}°')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide unused subplots
    for i in range(n_patterns, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Illumination Patterns', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def setup_publication_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })