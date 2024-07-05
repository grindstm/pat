import os
import sys
import signal
import yaml
import argparse

import numpy as np
from v_system.VSystemGenerator import VSystemGenerator
import scipy.ndimage
from perlin_numpy import generate_perlin_noise_3d

import jax.numpy as jnp
from jax import jit

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation
import util

def add_margin(image, N, margin, shift=(0, 0, 0)):
    """
    Place the image in the center of the domain, with margin and shift

    Parameters
    ----------
    image : ndarray
        The generated image
    N : tuple
        Size of the domain (3D)
    margin : ndarray
        Margin for each side of corresponding dimension. In other words, half the total margin in each dimension
    shift : ndarray
        Shift in each dimension

    Returns
    -------
    ndarray
        The image with the margin added
    """
    assert [N[i] + 2 * margin[i] + shift[i] == image.shape[i] for i in range(3)]

    image_out = jnp.zeros(N)
    image_out = image_out.at[
        margin[0] + shift[0] : N[0] - margin[0] + shift[0],
        margin[1] + shift[1] : N[1] - margin[1] + shift[1],
        margin[2] + shift[2] : N[2] - margin[2] + shift[2],
    ].set(image)

    return image_out


def point_plane(num_points, N, margin):
    """
    Generate points in the (x, y) plane at the z position with margins

    Parameters
    ----------
    num_points : int
        Number of points (use a number with an integer square root)
    margin : ndarray
        Size of the margin in each dimension.
    N : tuple
        Size of the domain (x, y, z)

    Returns
    -------
    tuple
        The positions
    """
    num_points_sqrt = jnp.sqrt(num_points).astype(int)
    x = jnp.linspace(margin[0], N[0] - margin[0], num_points_sqrt)
    y = jnp.linspace(margin[1], N[1] - margin[1], num_points_sqrt)
    z = jnp.ones(num_points) * (N[2] - margin[2])
    x, y = jnp.meshgrid(x, y)
    positions = (x.ravel(), y.ravel(), z)
    return positions


if __name__ == "__main__":
    # Signal handling
    def signal_handler(signum, frame):
        global exit_flag
        exit_flag = True
        print("Exit signal received, finishing current task...")

    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    # Parse parameters
    BATCH_SIZE, N, DX, C, CFL, PML_MARGIN, TISSUE_MARGIN, SENSOR_MARGIN, NUM_SENSORS, SOUND_SPEED_PERIODICITY, SOUND_SPEED_VARIATION_AMPLITUDE = util.parse_params()

    parser = argparse.ArgumentParser() 
    parser.add_argument("-o", type=str, default="data//", help="Output path")
    args = parser.parse_args()
    OUT_PATH = args.o
    # Parse arguments
    # if len(sys.argv) == 2:
    #     OUT_PATH = sys.argv[1]
    # else:
    #     OUT_PATH = "data/"

    # Output directories
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(f"{OUT_PATH}LNet/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}p0/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}sensors/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}p_data/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}c/", exist_ok=True)

    # ----------------------
    # Generate vessels
    tissue_margin = 2 * (np.array(TISSUE_MARGIN) + np.array([PML_MARGIN] * 3))
    tissue_volume = np.array(N) - tissue_margin
    print(f"Tissue volume: {tissue_volume}")
    shrink_factor = 2  # VSystemGenerator requires a minimum volume size to function properly
    sim = VSystemGenerator(tissue_volume=tissue_volume * shrink_factor)
    vessels_batch, n_iters = sim.create_networks(BATCH_SIZE)
    # shrink the vessels
    vessels_batch = [
        scipy.ndimage.zoom(vessels_batch[i], 1 / shrink_factor)
        for i in range(len(vessels_batch))
    ]
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

    # ----------------------
    # Set up the simulator
    domain = Domain(N, DX)
    sound_speed = jnp.ones(N) * C
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=PML_MARGIN)
    time_axis = TimeAxis.from_medium(medium, cfl=CFL)

    # ----------------------
    # Set up the sensors
    margin = np.array(SENSOR_MARGIN) + np.array([PML_MARGIN] * 3)
    sensor_positions = point_plane(NUM_SENSORS, N, margin)
    sensors_obj = BLISensors(sensor_positions, N)

    @util.timer
    @jit
    def compiled_simulator(p0, sound_speed):
        medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=PML_MARGIN)
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    for file in os.listdir(f"{OUT_PATH}LNet/"):
        if exit_flag:
            break

        print(f"Generating data for {file}")
        # The LNet files which don't have a corresponding p0 file
        if os.path.exists(OUT_PATH + f"p0/{file.split('_')[0]}"):
            continue
        file_index = file.split("_")[0]
        vessels = jnp.load(f"{OUT_PATH}LNet/{file}")

        # Generate Sound Speed
        if SOUND_SPEED_VARIATION_AMPLITUDE == 0:
            sound_speed = C * jnp.ones(N)
        else:
            sound_speed_volume = np.array(N)-np.array(PML_MARGIN)
            noise = generate_perlin_noise_3d(
                sound_speed_volume, [SOUND_SPEED_PERIODICITY] * 3, tileable=(False, False, False)
            )
            sound_speed = C + SOUND_SPEED_VARIATION_AMPLITUDE * noise
            sound_speed = add_margin(sound_speed, N, np.array(3*[PML_MARGIN // 2]), shift=(0, 0, -SENSOR_MARGIN[2]))
        c_file = f"{OUT_PATH}c/{file_index}.npy"
        jnp.save(c_file, sound_speed)

        # Add margin to the vessels
        p0 = add_margin(vessels, N, tissue_margin // 2, shift=(0, 0, -SENSOR_MARGIN[2]))
        p0_file = f"{OUT_PATH}p0/{file_index}.npy"
        jnp.save(p0_file, p0)

        p0 = jnp.expand_dims(p0, -1)
        p0 = FourierSeries(p0, domain)

        p_data = compiled_simulator(p0, sound_speed)

        p_data_file = f"{OUT_PATH}p_data/{file_index}.npy"
        jnp.save(p_data_file, p_data)
        sensors_file = f"{OUT_PATH}sensors/{file_index}.npy"
        jnp.save(sensors_file, sensor_positions)
        print(f"Saved {p0_file}, {p_data_file}, {sensors_file}, {c_file}")
