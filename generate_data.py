import os
import sys
import signal
import yaml

import numpy as np
from v_system.VSystemGenerator import VSystemGenerator

import jax.numpy as jnp
from jax import jit

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation


def generate_vessels(batch_size, N, pml_margin):
    """
    Generate a batch of Lindenmayer vessels

    Parameters
    ----------
    batch_size : int
        Number of vessels to generate
    N : tuple
        Size of the domain
    pml_margin : int
        Size of the PML

    Returns
    -------
    list, list : images, n_iters

    """
    sim = VSystemGenerator(
        n=batch_size,
        # d0_mean=2.0,
        # d0_std=.50,
        tissue_volume=[n - 2 * pml_margin for n in N],
    )

    return sim.create_networks()


def add_margin(image, N, margin):
    """
    Place the generated image in the center of the domain

    Parameters
    ----------
    image : ndarray
        The generated image
    N : tuple
        Size of the domain (3D)
    margin : int
        Size of the margin on each side

    Returns
    -------
    ndarray
        The image with the margin added
    """
    image_out = jnp.zeros(N)
    image_out = image_out.at[
        margin:-margin,
        margin:-margin,
        margin:-margin,
    ].add(image)
    return image_out


def sensor_plane(num_sensors, pml_margin, N, sensor_margin):
    """
    Generate sensor positions in a plane

    Parameters
    ----------
    num_sensors : int
        Number of sensors
    pml_margin : int
        Size of the PML
    N : tuple
        Size of the domain
    sensor_margin : int
        Additional margin within the PML

    Returns
    -------
    BLISensors
        The sensor object
    tuple
        The sensor positions
    """
    sensor_margin = pml_margin + sensor_margin
    num_sensors_sqrt = jnp.sqrt(num_sensors).astype(int)
    x = jnp.linspace(sensor_margin, N[0] - sensor_margin, num_sensors_sqrt)
    y = jnp.linspace(sensor_margin, N[1] - sensor_margin, num_sensors_sqrt)
    z = jnp.ones(num_sensors) * (N[2] - pml_margin)
    x, y = jnp.meshgrid(x, y)
    sensor_positions = (x.ravel(), y.ravel(), z)
    return BLISensors(positions=sensor_positions, n=N), sensor_positions


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
    PML_MARGIN = params["geometry"]["pml_margin"]
    DX = tuple(params["geometry"]["dx"])
    C = params["geometry"]["c"]
    CFL = params["geometry"]["cfl"]
    NUM_SENSORS = params["simulate"]["num_sensors"]
    SENSOR_MARGIN = params["simulate"]["sensor_margin"]
    BATCH_SIZE = params["generate_data"]["batch_size"]

    # Parse arguments
    if len(sys.argv) == 2:
        OUT_PATH = sys.argv[1]
    else:
        OUT_PATH = "data/"

    # Output directories
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(f"{OUT_PATH}LNet/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}p0/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}sensors/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}p_data/", exist_ok=True)

    # Generate vessels
    vessels_batch, n_iters = generate_vessels(BATCH_SIZE, N, PML_MARGIN)

    # Save vessels
    folder_index = (
        max(
            [
                int(filename.split("_")[0])
                for filename in os.listdir(f"{OUT_PATH}LNet/")
                if filename.split("_")[0].isdigit()
            ],
            default=-1,
        )
        + 1
    )
    print(folder_index)

    for i, vessels in enumerate(vessels_batch):
        filename = f"{i+folder_index}_{n_iters[i]}"

        np.save(f"{OUT_PATH}LNet/{filename}", vessels)
        print(f"Created vessels image {filename}")

    # Set up the simulator
    domain = Domain(N, DX)
    sound_speed = jnp.ones(N) * C
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=PML_MARGIN)
    time_axis = TimeAxis.from_medium(medium, cfl=CFL)

    # Set up the sensors
    sensors_obj, sensor_positions = sensor_plane(
        NUM_SENSORS, PML_MARGIN, N, SENSOR_MARGIN
    )

    @jit
    def compiled_simulator(p0):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    for file in os.listdir(f"{OUT_PATH}LNet/"):
        if exit_flag:
            break
        
        print(f"Processing {file}")
        # The LNet files which don't have a corresponding p0 file
        if os.path.exists(OUT_PATH + f"p0/{file.split('_')[0]}"):
            continue
        file_index = file.split("_")[0]
        vessels = jnp.load(f"{OUT_PATH}LNet/{file}")
        print(f"Generating data for {file}")

        # Add space for perfectly matched layer (PML) to the vessels
        p0 = add_margin(vessels, N, PML_MARGIN)
        p0_file = f"{OUT_PATH}p0/{file_index}.npy"
        jnp.save(p0_file, p0)

        p0 = jnp.expand_dims(p0, -1)
        p0 = FourierSeries(p0, domain)

        p_data = compiled_simulator(p0)


        p_data_file = f"{OUT_PATH}p_data/{file_index}.npy"
        jnp.save(p_data_file, p_data)
        sensors_file = f"{OUT_PATH}sensors/{file_index}.npy"
        jnp.save(sensors_file, sensor_positions)
        print(f"Saved {p0_file}, {p_data_file} and {sensors_file}")



# Reshape p_data to 3D
# p_data_3d = p_data.reshape(
#     int(time_axis.Nt),
#     int(jnp.sqrt(NUM_SENSORS)),
#     int(jnp.sqrt(NUM_SENSORS))
# )
# p_data_3d = jnp.transpose(p_data_3d, (1, 2, 0))

# Save p0, p_data and sensor positions