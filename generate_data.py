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
from jax import jit, vmap

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors, get_line_transducer
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
        Size of the domain (2D or 3D)
    margin : ndarray
        Margin for each side of corresponding dimension. In other words, half the total margin in each dimension
    shift : ndarray
        Shift in each dimension

    Returns
    -------
    ndarray
        The image with the margin added
    """
    # assert [N[i] + 2 * margin[i] + shift[i] == image.shape[i] for i in range(3)]
    if len(N) == 2:
        image_out = jnp.zeros(N)
        image_out = image_out.at[
            margin[0] + shift[0] : N[0] - margin[0] + shift[0],
            margin[1] + shift[1] : N[1] - margin[1] + shift[1],
        ].set(image)
        return image_out
    elif len(N) == 3:
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


def attenuation_mask_directional_3d(volume, azimuth_deg, elevation_deg, dx, mu):
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)

    ux = np.cos(elevation_rad) * np.cos(azimuth_rad)
    uy = np.cos(elevation_rad) * np.sin(azimuth_rad)
    uz = np.sin(elevation_rad)

    depth, height, width = volume.shape

    x_indices = np.arange(width)
    y_indices = np.arange(height)
    z_indices = np.arange(depth)
    X, Y, Z = np.meshgrid(x_indices, y_indices, z_indices, indexing="ij")

    distances = ux * X * dx + uy * Y * dx + uz * Z * dx

    mask = np.exp(-mu * distances)

    result = mask * volume
    return result


def generate_vessels_3d(N, batch_size, shrink_factor=1):
    sim = VSystemGenerator(tissue_volume=N * shrink_factor, d0_mean=20.0, d0_std=5.0)
    vessels_batch, n_iters = sim.create_networks(batch_size)
    if shrink_factor != 1:
        vessels_batch = [
            scipy.ndimage.zoom(vessels_batch[i], 1 / shrink_factor)
            for i in range(len(vessels_batch))
        ]
    return vessels_batch, n_iters


def apply_attenuation_masks(vessels, azimuth_deg, elevation_deg, dx, mu):
    """Applies directional attenuation masks to a batch of vessel data."""
    mask_vmap = vmap(
        attenuation_mask_directional_3d, in_axes=(0, None, None, None, None)
    )
    return mask_vmap(vessels, azimuth_deg, elevation_deg, dx, mu)


def process_vessels(
    dim,
    tissue_volume,
    batch_size,
    shrink_factor,
    lighting_attenuation,
    azimuth_deg,
    elevation_deg,
    dx,
    mu,
):
    """Generates and optionally applies attenuation to a batch of vessels."""
    vessels_batch, n_iters = generate_vessels_3d(
        tissue_volume, batch_size, shrink_factor
    )
    if lighting_attenuation:
        vessels_batch = apply_attenuation_masks(
            vessels_batch, azimuth_deg, elevation_deg, dx, mu
        )

    if dim == 2:
        vessels_batch = np.sum(vessels_batch, axis=1)

    return vessels_batch, n_iters


if __name__ == "__main__":
    # Signal handling
    def signal_handler(signum, frame):
        global exit_flag
        exit_flag = True
        print("Exit signal received, finishing current task...")

    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    # Parse parameters
    (
        BATCH_SIZE,
        N,
        SHRINK_FACTOR,
        DIMS,
        DX,
        C,
        CFL,
        PML_MARGIN,
        TISSUE_MARGIN,
        SENSOR_MARGIN,
        NUM_SENSORS,
        SOUND_SPEED_PERIODICITY,
        SOUND_SPEED_VARIATION_AMPLITUDE,
        LIGHTING_ATTENUATION,
        AZIMUTH_DEG,
        ELEVATION_DEG,
        MU,
    ) = util.parse_params()

    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str, default="data//", help="Output path")
    args = parser.parse_args()
    OUT_PATH = args.out_path

    # Output directories
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(f"{OUT_PATH}LNet/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}p0/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}sensors/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}p_data/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}c/", exist_ok=True)

    # ----------------------

    # Generate vessels
    tissue_margin = 2 * (np.array([PML_MARGIN] * 3) + np.array([PML_MARGIN] * 3))
    tissue_volume = np.array(N) - tissue_margin
    print(f"Tissue volume: {tissue_volume}")
    vessels_batch, n_iters = generate_vessels_3d(
        tissue_volume, BATCH_SIZE, shrink_factor=3
    )

    if DIMS == 2:
        OUT_PATH = os.path.join(OUT_PATH, "2d/")
        N = N[:2]
        DX = DX[:2]
    else:
        SHRINK_FACTOR = 2
    
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(f"{OUT_PATH}LNet/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}p0/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}sensors/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}p_data/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}c/", exist_ok=True)

    vessels_batch, n_iters = process_vessels(
        DIMS,
        tissue_volume,
        BATCH_SIZE,
        SHRINK_FACTOR,
        LIGHTING_ATTENUATION,
        AZIMUTH_DEG,
        ELEVATION_DEG,
        DX[0],
        MU,
    )

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

    # Set up the sensors
    if DIMS == 2:

        # sensor_positions = np.lin
        
        sensors_obj = get_line_transducer(domain, PML_MARGIN+SENSOR_MARGIN[0], N[1]-SENSOR_MARGIN[1]*2)
    elif DIMS == 3:
        print(DIMS)
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
            sound_speed_volume = np.array(N) - np.array(PML_MARGIN)
            noise = generate_perlin_noise_3d(
                sound_speed_volume,
                [SOUND_SPEED_PERIODICITY] * 3,
                tileable=(False, False, False),
            )
            sound_speed = C + SOUND_SPEED_VARIATION_AMPLITUDE * noise
            sound_speed = add_margin(
                sound_speed,
                N,
                np.array(3 * [PML_MARGIN // 2]),
                shift=(0, 0, -SENSOR_MARGIN[2]),
            )
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
