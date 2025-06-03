"""
Used to set jax environment variables, extract global variables from params.yamls, define plotting functions, ...
"""
import signal
import os
import time
import functools
import sys
import yaml
try:
    import colored_traceback.auto
except ImportError:
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import matplotlib.colors as mcolors


# environment variable to prevent jax preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"


def timer(func):
    """A decorator that prints the execution time of the function it decorates."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value, run_time

    return wrapper_timer


params = yaml.safe_load(open("params.yaml"))
DATA_PATH = params["file"]["data_path"]

BATCH_SIZE = params["generate_data"]["batch_size"]
N = tuple(params["generate_data"]["N"])
SHRINK_FACTOR = params["generate_data"]["shrink_factor"]
DIMS = params["generate_data"]["dims"]
DX = tuple(params["generate_data"]["dx"])
C = params["generate_data"]["c"]
C_PERIODICITY = params["generate_data"]["c_periodicity"]
C_VARIATION_AMPLITUDE = params["generate_data"]["c_variation_amplitude"]
C_BLOOD = params["generate_data"]["c_blood"]
CFL = params["generate_data"]["cfl"]
PML_MARGIN = params["generate_data"]["pml_margin"]
TISSUE_MARGIN = params["generate_data"]["tissue_margin"]
SENSOR_MARGIN = tuple(params["generate_data"]["sensor_margin"])
NUM_SENSORS = params["generate_data"]["num_sensors"]
NOISE_AMPLITUDE = params["generate_data"]["noise_amplitude"]

LIGHTING_ATTENUATION = params["lighting"]["lighting_attenuation"]
NUM_LIGHTING_ANGLES = params["lighting"]["num_lighting_angles"]
ATTENUATION = params["lighting"]["attenuation"]

RECON_ITERATIONS = params["reconstruct"]["recon_iterations"]
LR_MU_R = params["reconstruct"]["lr_mu_r"]
LR_C_R = params["reconstruct"]["lr_c_r"]
RECON_FILE_START = params["reconstruct"]["recon_file_start"]
RECON_FILE_END = params["reconstruct"]["recon_file_end"]

CHECKPOINT_FILE_INDEX = params["train"]["checkpoint_index"]
LR_R_MU = params["train"]["lr_R_mu"]
LR_R_C = params["train"]["lr_R_c"]
DROPOUT = params["train"]["dropout"]
TRAIN_FILE_START = params["train"]["train_file_start"]
TRAIN_FILE_END = params["train"]["train_file_end"]

mu_path = os.path.join(DATA_PATH, "mu")
angles_path = os.path.join(DATA_PATH, "angles")
ATT_masks_path = os.path.join(DATA_PATH, "ATT_masks")
p0_path = os.path.join(DATA_PATH, "P_0")
c_path = os.path.join(DATA_PATH, "c")
sensors_path = os.path.join(DATA_PATH, "sensors")
P_0_path = os.path.join(DATA_PATH, "P_0")
P_data_path = os.path.join(DATA_PATH, "P_data")
P_data_noisy_path = os.path.join(DATA_PATH, "P_data_noisy")
mu_r_path = os.path.join(DATA_PATH, "mu_r")
c_r_path = os.path.join(DATA_PATH, "c_r")
params_R_mu_path = os.path.join(DATA_PATH, "checkpoints", "params_R_mu")
params_R_c_path = os.path.join(DATA_PATH, "checkpoints", "params_R_c")
checkpoints_path = os.path.abspath(os.path.join(DATA_PATH, "checkpoints", str(CHECKPOINT_FILE_INDEX)))
state_path = os.path.join(DATA_PATH, "state")
profile_dir = os.path.join(DATA_PATH, "profile")
pres_path = os.path.abspath('../../Presentation/figures/')

os.makedirs(DATA_PATH, exist_ok=True)

os.makedirs(mu_path, exist_ok=True)
os.makedirs(angles_path, exist_ok=True)
os.makedirs(c_path, exist_ok=True)
os.makedirs(P_data_path, exist_ok=True)
os.makedirs(P_data_noisy_path, exist_ok=True)
os.makedirs(sensors_path, exist_ok=True)
os.makedirs(P_0_path, exist_ok=True)
os.makedirs(mu_r_path, exist_ok=True)
os.makedirs(c_r_path, exist_ok=True)
os.makedirs(ATT_masks_path, exist_ok=True)
os.makedirs(profile_dir, exist_ok=True)


def file(path, index, iteration=None):
    if iteration is not None:
        return os.path.join(path, f"{index}_{iteration}.npy")
    else:
        return os.path.join(path, f"{index}.npy")


def max_file_index(path):
    return (
        max(
            [
                int(filename.split("_")[0])
                for filename in os.listdir(path)
                if filename.split("_")[0].isdigit()
            ],
            default=-1,
        )
        + 1
    )


def max_iteration():
    """
    Returns the maximum iteration number in the mu_r_path directory.
    """
    if not os.path.exists(mu_r_path):
        return 0
    else:
        return max(
            [
                int(filename.split("_")[1].split(".")[0])
                for filename in os.listdir(mu_r_path)
                if filename.split("_")[1].split(".")[0].isdigit()
            ],
            default=0,
        )
    





# -------------------------
# Plotting
# -------------------------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

def create_colormap(data, gt):
    """
    Usage example:
        cmap_mu_r, norm_mu_r = create_colormap(mu_r, mu)
        cmap_c_r, norm_c_r = create_colormap(c_r, c)
    """
    n_colors = 256
    vmin_data, vmax_data = np.min(data), np.max(data)
    vmin_gt, vmax_gt = np.min(gt), np.max(gt)

    vmax = max(vmax_data, vmax_gt)
    vmin = min(vmin_data, vmin_gt)
    vrange = vmax - vmin

    q = (vmax_gt - vmin_gt) / vrange 
    n_q = int(q * n_colors)
    colors_in_range = plt.cm.viridis(np.linspace(0., 1, n_q)) 
    colors = [colors_in_range]
    if vmax_data>vmax_gt:
        r = (vmax_data - vmax_gt) / vrange
        n_r = int(r * n_colors)
        colors_above = plt.cm.YlOrBr(np.linspace(0+.4, r+.4, n_r)) 
        colors.append(colors_above)

    if vmin_data<vmin_gt:
        p = (vmin_gt - vmin_data) / vrange
        n_p = int(p * n_colors)
        colors_below = plt.cm.BuPu(np.linspace(0+.6, p+.6, n_p))
        colors.insert(0, colors_below)

    colors = np.vstack(colors)
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm

def plot_recon_animation(mu_rs, c_rs, mu, c, save_path, repeat_delay=1000):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.05, 1, 1, 0.05], height_ratios=[1, 1])

    # Axes for the images
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    # Axes for the colorbars
    cax1 = fig.add_subplot(gs[0, 0]) 
    cax2 = fig.add_subplot(gs[1, 0]) 
    cax3 = fig.add_subplot(gs[0, 3]) 
    cax4 = fig.add_subplot(gs[1, 3]) 

    # colormaps
    cmap_mu_r, norm_mu_r = create_colormap(mu_rs, mu)
    cmap_c_r, norm_c_r = create_colormap(c_rs, c)

    ims = []
    num_it = len(mu_rs)
    for i in range(num_it):
        title = ax1.text(1.1, 1.1, fr'Iteration {i}', size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax1.transAxes, animated=True)

        im0 = ax1.imshow(mu_rs[i], cmap=cmap_mu_r, norm=norm_mu_r, animated=True)
        im1 = ax2.imshow(c_rs[i], cmap=cmap_c_r, norm=norm_c_r, animated=True)

        im2 = ax3.imshow(mu, cmap='viridis', animated=True)
        im3 = ax4.imshow(c, cmap='viridis', animated=True)
        
        ims.append([im0, im1, im2, im3, title])

    if repeat_delay:
        for _ in range(6): 
            ims.append([im0, im1, im2, im3, title])

    cbar_mu_r = plt.colorbar(im0, cax=cax1)
    cbar_c_r = plt.colorbar(im1, cax=cax3)
    plt.colorbar(im2, cax=cax2)
    plt.colorbar(im3, cax=cax4)

    ax1.set_title('mu_r')
    ax2.set_title('c_r')
    ax3.set_title('mu (GT)')
    ax4.set_title('c (GT)')

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat=bool(repeat_delay), repeat_delay=repeat_delay)
    ani.save(save_path, writer='ffmpeg', fps=6)

    plt.show()

    