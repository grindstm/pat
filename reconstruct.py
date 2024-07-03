import sys
import os
import signal
import yaml

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax import random


from jwave import FourierSeries
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation
from jwave.signal_processing import smooth

import time
import functools

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
    """
    noise = random.normal(random.PRNGKey(key), p_data.shape)
    for i in range(noise.shape[1]):
        noise = noise.at[:, i].set(smooth(noise[:, i],blackman_window_exponent))
    return p_data + amplitude * noise

if __name__ == "__main__":
    # Signal handling
    def signal_handler(signum, frame):
        global exit_flag
        exit_flag = True
        print("Exit signal received, finishing current task...")

    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    # Parse parameters
    params = yaml.safe_load(open("params.yaml"))
    N = tuple(params["geometry"]["N"])
    DX = tuple(params["geometry"]["dx"])
    C = params["geometry"]["c"]
    CFL = params["geometry"]["cfl"]
    PML_MARGIN = params["geometry"]["pml_margin"]
    TISSUE_MARGIN = params["geometry"]["tissue_margin"]
    SENSOR_MARGIN = params["geometry"]["sensor_margin"]
    NUM_SENSORS = params["geometry"]["num_sensors"]

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

    # Set up the simulator
    domain = Domain(N, DX)
    sound_speed = jnp.ones(N) * C
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=PML_MARGIN)
    time_axis = TimeAxis.from_medium(medium, cfl=CFL)

    def simulator(p0, sensors_obj):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)


    def mse_loss(p0, p_data, sensors_obj):
        p0 = p0.replace_params(p0.params)
        p_pred = simulator(p0,sensors_obj=sensors_obj)[..., 0]
        return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2)

    @jit
    def lazy_time_reversal(p0, p_data, sensors_obj):

        p0 = FourierSeries.empty(domain)

        p_grad = grad(mse_loss)(p0, p_data, sensors_obj)

        return -p_grad

    
    @util.timer
    def reconstruct(file):
        file_index = file.split(".")[0]

        # Set up the sensors
        sensor_positions = jnp.load(f"{OUT_PATH}sensors/{file}")
        sensors_obj = BLISensors(positions=sensor_positions, n=N)

        p0_file = f"{IN_PATH}p0/{file_index}.npy"
        print(p0_file)
        p0 = jnp.load(p0_file)
        p_data_file = f"{IN_PATH}p_data/{file_index}.npy"
        p_data = jnp.load(p_data_file)

        # Add noise to p_data
        k0, k1 = jax.random.PRNGKey(int(time.time()))
        p_data_noisy = add_colored_noise(k1, p_data)
    
        p_r = lazy_time_reversal(p0, p_data_noisy, sensors_obj)

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

