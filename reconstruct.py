import os
import signal
import argparse
import time

import jax
import jax.numpy as jnp
from jax import random

from jwave.signal_processing import smooth

import time_reversal
import util as u

from jax import lax
def clear_jax_cache():
    lax.clear_caches()


def add_colored_noise(key, data, blackman_window_exponent=1, amplitude=0.2):
    """
    Add colored noise to the data

    Parameters
    ----------
    key : PRNGKey
        Random key
    data : ndarray
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
    noise = random.normal(key, data.shape)
    for i in range(noise.shape[1]):
        noise = noise.at[:, i].set(smooth(noise[:, i], blackman_window_exponent))
    return data + amplitude * noise

add_colored_noise_vmap = jax.vmap(add_colored_noise, in_axes=(0, 0, None, None))

if __name__ == "__main__":
    # Signal handling
    def signal_handler(signum, frame):
        global exit_flag
        exit_flag = True
        print("Exit signal received, finishing current task...")

    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", type=str, default=None, help="data path", nargs="?"
    )
    args = parser.parse_args()
    if args.data_path is not None:
        u.DATA_PATH = args.data_path

    os.makedirs(u.DATA_PATH, exist_ok=True)
    os.makedirs(u.DATA_PATH + "p_r/", exist_ok=True)
    os.makedirs(u.DATA_PATH + "c_r/", exist_ok=True)

    @u.timer
    def reconstruct(file):
        # print current wall time
        print(time.time())

        file_index = file.split(".")[0]

        # Set up the sensors
        # ----------------------
        sensor_positions = jnp.load(f"{u.DATA_PATH}sensors/{file}")

        # Load the data
        # ----------------------
        p_data_file = f"{u.DATA_PATH}p_data/{file_index}.npy"
        p_data = jnp.load(p_data_file)
        angles_file = f"{u.DATA_PATH}angles/{file_index}.npy"
        angles = jnp.load(angles_file)
        
        # p0_file = f"{u.DATA_PATH}p0/{file_index}.npy"
        # p0 = jnp.load(p0_file)
        # print(p0_file)

        # c_file = f"{u.DATA_PATH}c/{file_index}.npy"
        # c = jnp.load(c_file)

        # Add noise to p_data
        # ----------------------
        key = jax.random.PRNGKey(int(time.time()))
        keys = jax.random.split(key, u.NUM_LIGHTING_ANGLES)
        add_noise_vmap = jax.vmap(add_colored_noise, in_axes=(0, 0, None))

        p_data_noisy = add_noise_vmap(keys, p_data, u.NOISE_AMPLITUDE)
 
        # p_data_noisy = add_colored_noise(k1, p_data, amplitude=u.NOISE_AMPLITUDE)
        # ----------------------------------------
        p_rs, c_rs, mses = time_reversal.multi_illumination(p_data_noisy, sensor_positions, angles, num_iterations=u.NUM_ITERATIONS, learning_rate=u.LEARNING_RATE)
 
        # p_r = time_reversal.lazy_time_reversal(p_data_noisy, sensor_positions)
        # p_rs = [p_r.on_grid]

        # p_rs, mses = time_reversal.iterative_time_reversal(p_data_noisy, sensor_positions, num_iterations=NUM_ITERATIONS, learning_rate=LEARNING_RATE)
        # print(f"Mean squared errors: {mses}")

        # p_r, losses = time_reversal.iterative_time_reversal_optimized(p_data_noisy, sensor_positions, num_iterations=4)
        # print(f"Losses: {losses}")
        # ----------------------------------------
        # print(len(p_rs))

        for i, p_r in enumerate(p_rs):
            if exit_flag:
                break
            p_r_file = f"{u.DATA_PATH}p_r/{file_index}_{i}.npy"
            jnp.save(p_r_file, p_r.squeeze())
            print(f"Saved {p_r_file}")

            c_r_file = f"{u.DATA_PATH}c_r/{file_index}_{i}.npy"
            jnp.save(c_r_file, c_rs[i].squeeze())
            print(f"Saved {c_r_file}")

    for file in os.listdir(f"{u.DATA_PATH}p0/"):
        if exit_flag:
            break

        print(f"Processing {file}")
        # p0 files which don't have a corresponding p_r file
        if os.path.exists(u.DATA_PATH + f"p_r/{file.split('.')[0]}"):
            continue

        reconstruct(file)
