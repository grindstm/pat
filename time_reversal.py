import sys
import os
import jax.numpy as jnp
import jax
from jwave.utils import show_field
import yaml
from simulator import Simulator
from jax import grad, jit
from jwave import FourierSeries

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation


def add_gaussian_noise(data, noise_std_dev, key):
    noise = jax.random.normal(key, shape=data.shape, dtype=data.dtype) * noise_std_dev
    return data + noise


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    N = tuple(params["geometry"]["N"])
    PML_MARGIN = params["geometry"]["pml_margin"]
    DX = tuple(params["geometry"]["dx"])
    C = params["geometry"]["c"]
    CFL = params["geometry"]["cfl"]
    NUM_SENSORS = params["simulate"]["num_sensors"]
    SENSOR_MARGIN = params["simulate"]["sensor_margin"]

    if len(sys.argv) == 3:
        IN_PATH = sys.argv[1]
        OUT_PATH = sys.argv[2]
    else:
        IN_PATH = "data/"
        OUT_PATH = "data/"

    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(f"{OUT_PATH}p_r/", exist_ok=True)

    # Set up the simulator
    domain = Domain(N, DX)
    sound_speed = jnp.ones(N) * C
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=PML_MARGIN)
    time_axis = TimeAxis.from_medium(medium, cfl=CFL)

    # Set up the sensors
    def simulator(p0):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    @jit
    def lazy_time_reversal(p0, p_data):
        def mse_loss(p0, p_data):
            p0 = p0.replace_params(p0.params)
            p_pred = simulator(p0)[..., 0]
            return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2)

        p0 = FourierSeries.empty(domain)

        p_grad = grad(mse_loss)(p0, p_data)

        return -p_grad

    for file in os.listdir(f"{IN_PATH}p0/"):
        print(f"Processing {file}")
        # p0 files which don't have a corresponding p_r file
        if os.path.exists(OUT_PATH + f"p_r/{file.split('.')[0]}"):
            continue
        file_index = file.split(".")[0]

        sensor_positions = jnp.load(f"{OUT_PATH}sensors/{file}")
        sensors_obj = BLISensors(positions=sensor_positions, n=N)

        p0_file = f"{IN_PATH}p0/{file_index}.npy"
        p0 = jnp.load(p0_file)
        p_data_file = f"{IN_PATH}p_data/{file_index}.npy"
        p_data = jnp.load(p_data_file)

        # Add Gaussian noise to p_data
        key = jax.random.PRNGKey(seed=0)
        noise_std_dev = 0.05  # Standard deviation of the Gaussian noise
        key, subkey = jax.random.split(key)  # Split the key for reuse
        p_data_noisy = add_gaussian_noise(p_data, noise_std_dev, subkey)

        # visualize p0_noisy
        # show_field(p_data_noisy, "Section of the initial pressure")
        
        p_r = lazy_time_reversal(p0, p_data_noisy)

        p_r_file = f"{OUT_PATH}p_r/{file_index}.npy"
        jnp.save(p_r_file, p_r.on_grid)
        print(f"Saved {p_r_file}")


        # ----------------------------