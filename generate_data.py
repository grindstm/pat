import os
import signal
import argparse

import numpy as np
from v_system.VSystemGenerator import VSystemGenerator
import scipy.ndimage
from perlin_numpy import generate_perlin_noise_3d, generate_perlin_noise_2d

import jax.numpy as jnp
from jax import jit, vmap

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation
import util

import jax

# __ CPU
from jax import device_put, devices

# __ map
import jax.lax as lax
from functools import partial

jax.clear_caches()


def add_margin(image, N, margin, shift):
    """
    Place the image in the center of the domain, with margin and shift.

    Parameters
    ----------
    image : ndarray
        The generated image (2D or 3D).
    N : tuple
        Size of the domain (2D or 3D).
    margin : ndarray
        Margin for each side of corresponding dimension.
    shift : ndarray
        Shift in each dimension.

    Returns
    -------
    ndarray
        The image with the margin added.
    """
    image_out = jnp.zeros(N)
    slices = tuple(
        slice(m + s, dim_size - m + s) for m, s, dim_size in zip(margin, shift, N)
    )
    image_out = image_out.at[slices].set(image)
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


@jit
def attenuation_mask_directional_3d(angles, volume, dx, mu):
    azimuth_deg, elevation_deg = angles[0], angles[1]
    azimuth_rad = jnp.deg2rad(azimuth_deg)
    elevation_rad = jnp.deg2rad(elevation_deg)

    ux = jnp.cos(elevation_rad) * jnp.cos(azimuth_rad)
    uy = jnp.cos(elevation_rad) * jnp.sin(azimuth_rad)
    uz = jnp.sin(elevation_rad)

    depth, height, width = volume.shape
    x_indices = jnp.arange(width)
    y_indices = jnp.arange(height)
    z_indices = jnp.arange(depth)
    X, Y, Z = jnp.meshgrid(x_indices, y_indices, z_indices, indexing="ij")

    distances = ux * X * dx + uy * Y * dx + uz * Z * dx
    mask = jnp.exp(-mu * distances)
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
    return np.array(vessels_batch), n_iters


def illuminate_vessels(
    tissue_volume,
    mu,
    dx,
    azimuth_degs,
    elevation_degs,
):
    """
    Returns a batch of p0 images for a volume of tissue illuminated at given angles.

    Parameters
    ----------
    tissue_volume : ndarray
        The voxel volume of the tissue
    mu : float
        The attenuation coefficient
    dx : float
        The voxel size
    azimuth_deg : array
        The azimuth angles in degrees
    elevation_deg : array
        The elevation angles in degrees

    Returns
    -------
    ndarray
        The batch of illuminated vessel images
    """

    # __ vmap # memory error
    # attenuate_vmap = vmap(
    #     attenuation_mask_directional_3d, in_axes=(None, 0, 0, None, None)
    # )

    # attenuated_vessels = attenuate_vmap(
    #     tissue_volume, azimuth_degs, elevation_degs, dx, mu
    # )

    # __ map
    angles = jnp.stack([azimuth_degs, elevation_degs], axis=1)

    partial_attenuation_mask = partial(
        attenuation_mask_directional_3d, volume=tissue_volume, dx=dx, mu=mu
    )
    attenuated_vessels = lax.map(partial_attenuation_mask, angles)

    return attenuated_vessels


if __name__ == "__main__":
    # Signal handling
    def signal_handler(signum, frame):
        global exit_flag
        exit_flag = True
        print("Exit signal received, finishing current task...")

    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    # These variables are loaded into this namespace by set_globals and
    # these definitions are here to keep the python language server happy
    (
        N,
        BATCH_SIZE,
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
        NUM_LIGHTING_ANGLES,
        MU,
    ) = (
        (None, None, None),
        None,
        None,
        None,
        (None, None, None),
        None,
        None,
        None,
        (None, None, None),
        (None, None, None),
        None,
        None,
        None,
        None,
        None,
        None,
    )

    util.set_globals()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", type=str, default=None, help="data path", nargs="?"
    )
    args = parser.parse_args()
    if args.data_path is not None:
        DATA_PATH = args.data_path

    os.makedirs(DATA_PATH, exist_ok=True)

    # ----------------------
    # Generate vessels
    # ----------------------
    print("Generating vessels")
    tissue_margin = 2 * (np.array([PML_MARGIN] * 3) + np.array([PML_MARGIN] * 3))
    tissue_volume = np.array(N) - tissue_margin
    print(f"Tissue volume: {tissue_volume}")
    vessels_batch, n_iters = generate_vessels_3d(
        tissue_volume, BATCH_SIZE, shrink_factor=SHRINK_FACTOR
    )

    # Save vessels
    # ----------------------
    os.makedirs(f"{DATA_PATH}LNet/", exist_ok=True)
    folder_index = (
        max(
            [
                int(filename.split("_")[0])
                for filename in os.listdir(f"{DATA_PATH}LNet/")
                if filename.split("_")[0].isdigit()
            ],
            default=-1,
        )
        + 1
    )

    for i, vessels in enumerate(vessels_batch):
        filename = f"{i+folder_index}_{n_iters[i]}"

        lnet_file = f"{DATA_PATH}LNet/{filename}.npy"
        np.save(lnet_file, vessels)
        print(f"Created vessels image {lnet_file}")

    # ----------------------
    # ----------------------
    # Simulation setup
    # ----------------------
    if DIMS == 2:
        N2 = tuple(N[:2])
        DX2 = tuple(DX[:2])
        domain = Domain(N2, DX2)

        # Set up the sensors
        x_start = SENSOR_MARGIN[1] + PML_MARGIN
        x_end = N[1] - SENSOR_MARGIN[1] - PML_MARGIN
        x = np.linspace(x_start, x_end, NUM_SENSORS)
        y = np.ones_like(x) * (SENSOR_MARGIN[0] + PML_MARGIN)
        sensor_positions = np.array([x, y])
        sensors_obj = BLISensors(sensor_positions, domain.N)

    elif DIMS == 3:
        domain = Domain(N, DX)

        # Set up the sensors
        margin = np.array(SENSOR_MARGIN) + np.array([PML_MARGIN] * 3)
        sensor_positions = point_plane(NUM_SENSORS, N, margin)
        sensors_obj = BLISensors(sensor_positions, N)

    @util.timer
    @jit
    def compiled_simulator(medium, time_axis, p0):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    batch_compiled_simulator = vmap(compiled_simulator, in_axes=(None, None, 0))

    os.makedirs(f"{DATA_PATH}angles/", exist_ok=True)
    os.makedirs(f"{DATA_PATH}sensors/", exist_ok=True)
    os.makedirs(f"{DATA_PATH}p0/", exist_ok=True)
    os.makedirs(f"{DATA_PATH}p_data/", exist_ok=True)
    os.makedirs(f"{DATA_PATH}c/", exist_ok=True)
    # Loop through the LNet files which don't have a corresponding p0 file
    # ----------------------
    for file in os.listdir(f"{DATA_PATH}LNet/"):
        if exit_flag:
            break

        if os.path.exists(DATA_PATH + f"p0/{file.split('_')[0]}*"):
            continue
        file_index = file.split("_")[0]

        # Illumination
        # ----------------------
        vessels = jnp.load(f"{DATA_PATH}LNet/{file}")

        # Generate random illumination angles
        azimuth_degs = np.random.uniform(0, 360, NUM_LIGHTING_ANGLES)
        elevation_degs = np.random.uniform(0, 180, NUM_LIGHTING_ANGLES)

        # Save the illumination angles
        angles_file = f"{DATA_PATH}angles/{file_index}.npy"
        jnp.save(angles_file, np.array([azimuth_degs, elevation_degs]))

        attenuated_vessels = illuminate_vessels(
            vessels, MU, DX[0], azimuth_degs, elevation_degs
        )

        # __ CPU
        # vessels_cpu = device_put(vessels, devices("cpu")[0])

        # attenuated_vessels = jit(illuminate_vessels, backend="cpu")(
        #     vessels_cpu, MU, DX[0], azimuth_degs, elevation_degs
        # )

        # Add margin to the vessels
        # ----------------------
        p0 = []
        for i in range(attenuated_vessels.shape[0]):
            p0.append(
                device_put(
                    add_margin(
                        attenuated_vessels[i],
                        N,
                        tissue_margin // 2,
                        shift=(0, 0, -1 * SENSOR_MARGIN[2]),
                    ),
                    devices("cpu")[0],
                )
            )

        # __ map
        # partial_add_margin = partial(
        #     add_margin, N=N, margin=tissue_margin // 2, shift=(0, 0, -SENSOR_MARGIN[2])
        # )
        # p0 = lax.map(partial_add_margin, attenuated_vessels)

        #  __ vmap
        # p0 = vmap(add_margin, in_axes=(0, None, None, None))(
        #     attenuated_vessels, N, tissue_margin // 2, (0, 0, -SENSOR_MARGIN[2])
        # )

        # __ CPU
        # attenuated_vessels_cpu = device_put(attenuated_vessels, devices("cpu")[0])

        # p0 = vmap(add_margin, in_axes=(0, None, None, None))(
        #     attenuated_vessels_cpu, N, tissue_margin // 2, (0, 0, -SENSOR_MARGIN[2])
        # )

        # Sound speed
        # ----------------------
        if DIMS == 2:
            # flatten the images
            p0 = jnp.sum(jnp.array(p0), axis=1)

            # Generate Sound Speed
            if SOUND_SPEED_VARIATION_AMPLITUDE == 0:
                sound_speed = C * jnp.ones(N2)
            else:
                # sound_speed_area = np.array(N2) - np.array(PML_MARGIN)
                noise = generate_perlin_noise_2d(
                    N2,
                    # sound_speed_area,
                    [SOUND_SPEED_PERIODICITY] * 2,
                    tileable=(False, False),
                )
                sound_speed = C + SOUND_SPEED_VARIATION_AMPLITUDE * noise
            # sound_speed = add_margin(
            #     sound_speed,
            #     N2,
            #     np.array(2 * [PML_MARGIN // 2]),
            #     shift=(-SENSOR_MARGIN[0], 0),
            # )

        elif DIMS == 3:
            # Generate Sound Speed
            if SOUND_SPEED_VARIATION_AMPLITUDE == 0:
                sound_speed = C * jnp.ones(N)
            else:
                # sound_speed_volume = np.array(N) - np.array(PML_MARGIN)
                noise = generate_perlin_noise_3d(
                    N,
                    # sound_speed_volume,
                    [SOUND_SPEED_PERIODICITY] * 3,
                    tileable=(False, False, False),
                )
                sound_speed = C + SOUND_SPEED_VARIATION_AMPLITUDE * noise
            # sound_speed = add_margin(
            #     sound_speed,
            #     N,
            #     np.array(3 * [PML_MARGIN // 2]),
            #     shift=(0, 0, -SENSOR_MARGIN[2]),
            # )

        # Save p0
        # ----------------------
        p0_file = f"{DATA_PATH}p0/{file_index}.npy"
        jnp.save(p0_file, p0)

        # Save sound speed
        # ----------------------
        c_file = f"{DATA_PATH}c/{file_index}.npy"
        jnp.save(c_file, sound_speed)

        # ----------------------

        # ----------------------
        # Perform simulation
        # ----------------------
        print(f"Simulating {file}")
        # print(sound_speed.shape, sound_speed.max(), sound_speed.min())
        sound_speed = FourierSeries(jnp.expand_dims(sound_speed, -1), domain)
        medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=PML_MARGIN)
        time_axis = TimeAxis.from_medium(medium, cfl=CFL)
        # print(time_axis.dt, time_axis.Nt)
        p0 = device_put(p0, devices("cuda")[0])
        p0 = vmap(FourierSeries, (0, None))(p0, domain)

        p_data = batch_compiled_simulator(medium, time_axis, p0)

        # ----------------------

        # ----------------------
        # Save the data
        # # ----------------------
        sensors_file = f"{DATA_PATH}sensors/{file_index}.npy"
        jnp.save(sensors_file, sensor_positions)
        p_data_file = f"{DATA_PATH}p_data/{file_index}.npy"
        jnp.save(p_data_file, p_data)

        print(f"Saved {p0_file}, {p_data_file}, {sensors_file}, {c_file}")
