import sys
import os
import signal

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax import random

from jwave.signal_processing import smooth

import time
import time_reversal

import util


def add_colored_noise(key, p_data, blackman_window_exponent=1, amplitude=0.2):
    """
    Add colored noise to the data

    Parameters
    ----------
    key : PRNGKey
        Random key
    p_data : ndarray
        Pressure data
    blackman_window_exponent : float
        Exponent of the Blackman window
    amplitude : float
        Amplitude of the noise

    Returns
    -------
    ndarray

    References
    ----------
    https://ucl-bug.github.io/jwave/notebooks/ivp/homogeneous_medium_backprop.html
    """
    noise = random.normal(random.PRNGKey(key), p_data.shape)
    for i in range(noise.shape[1]):
        noise = noise.at[:, i].set(smooth(noise[:, i], blackman_window_exponent))
    return p_data + amplitude * noise


if __name__ == "__main__":
    # Signal handling
    def signal_handler(signum, frame):
        global exit_flag
        exit_flag = True
        print("Exit signal received, finishing current task...")

    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    # Parse arguments
    if len(sys.argv) == 2:
        IN_PATH = sys.argv[1]
        OUT_PATH = sys.argv[1]
    elif len(sys.argv) == 3:
        IN_PATH = sys.argv[1]
        OUT_PATH = sys.argv[2]
    else:
        IN_PATH = "data/"
        OUT_PATH = "data/"

    # Output directories
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(f"{OUT_PATH}p_r/", exist_ok=True)

    @util.timer
    def reconstruct(file):
        # print current wall time
        print(time.time())

        file_index = file.split(".")[0]

        # # Set up the sensors
        sensor_positions = jnp.load(f"{OUT_PATH}sensors/{file}")

        p0_file = f"{IN_PATH}p0/{file_index}.npy"
        print(p0_file)
        p0 = jnp.load(p0_file)
        p_data_file = f"{IN_PATH}p_data/{file_index}.npy"
        p_data = jnp.load(p_data_file)

        # Add noise to p_data
        k0, k1 = jax.random.PRNGKey(int(time.time()))
        p_data_noisy = add_colored_noise(k1, p_data)

        # p_r = time_reversal.lazy_time_reversal(p0, p_data_noisy, sensor_positions)

        p_r, mses = time_reversal.iterative_time_reversal(p0, p_data_noisy, sensor_positions, num_iterations=4, learning_rate=50.) 
        print(f"Mean squared errors: {mses}")
        
        # p_r, losses = time_reversal.iterative_time_reversal_optimized(p0, p_data_noisy, sensor_positions, num_iterations=4)
        # print(f"Losses: {losses}")



        p_r_file = f"{OUT_PATH}p_r/{file_index}.npy"
        jnp.save(p_r_file, p_r.on_grid)
        print(f"Saved {p_r_file}")

    for file in os.listdir(f"{IN_PATH}p0/"):
        if exit_flag:
            break

        print(f"Processing {file}")
        # p0 files which don't have a corresponding p_r file
        if os.path.exists(OUT_PATH + f"p_r/{file.split('.')[0]}"):
            continue

        reconstruct(file)
