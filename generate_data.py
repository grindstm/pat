import os
import signal
import argparse

import numpy as np
from v_system.VSystemGenerator import VSystemGenerator
import scipy.ndimage
from perlin_numpy import generate_perlin_noise_3d, generate_perlin_noise_2d

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation

import jax
import jax.numpy as jnp
from jax import jit, vmap

# __ CPU
from jax import device_put, devices

# __ map
import jax.lax as lax
from functools import partial

import util as u

jax.clear_caches()


def generate_vessels_3d(N, batch_size, shrink_factor=1):
    """
    Generate a batch of 3D vessel images.

    Parameters
    ----------
    N : tuple
        Size of the domain (x, y, z).
    batch_size : int
        Number of images to generate.
    shrink_factor : int
        Images are generated at a larger size, then shrunk by this factor to mitigate edge effects.
        tissue volumes lower than about 128x128x128 benefit from a shrink factor of 2.

    Returns
    -------
    ndarray
        The batch of generated images.
    int
        The number of iterations used to generate the images.
    """
    sim = VSystemGenerator(tissue_volume=N * shrink_factor, d0_mean=20.0, d0_std=5.0)
    vessels_batch, n_iters = sim.create_networks(batch_size)
    if shrink_factor != 1:
        vessels_batch = [
            scipy.ndimage.zoom(vessels_batch[i], 1 / shrink_factor)
            for i in range(len(vessels_batch))
        ]
    return np.array(vessels_batch), n_iters


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


import jax.numpy as jnp
from jax import jit


# @jit
def attenuation_mask_directional_2d(angle, volume, dx, mu):
    """
    Compute the attenuation mask for a 2D volume given an angle, voxel size, and attenuation coefficient.

    Parameters
    ----------
    angle : float
        The direction angle in degrees.
    volume : ndarray
        The 2D volume to be attenuated.
    dx : float
        The voxel size.
    mu : float
        The attenuation coefficient.

    Returns
    -------
    ndarray
        The attenuated 2D volume.
    """
    angle_rad = jnp.deg2rad(angle)

    ux = jnp.cos(angle_rad)
    uy = jnp.sin(angle_rad)

    height, width = volume.shape
    x_indices = jnp.arange(width)
    y_indices = jnp.arange(height)
    X, Y = jnp.meshgrid(x_indices, y_indices, indexing="ij")

    distances = ux * X * dx + uy * Y * dx
    mask = jnp.exp(-mu * distances)
    result = mask * volume
    return result


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


def batch_attenuate_light(volume, mu, dx, angles):
    """
    Returns a batch of images for a volume illuminated at given angles.

    Parameters
    ----------
    volume : ndarray
        The voxel volume of the tissue
    mu : float
        The attenuation coefficient
    dx : float
        The voxel size
    angles : ndarray
        The angles of illumination (azimuth and elevation)

    Returns
    -------
    ndarray
        The batch of illuminated images
    """

    partial_attenuation_mask = partial(
        attenuation_mask_directional_3d, volume=volume, dx=dx, mu=mu
    )
    attenuated_volume = lax.map(partial_attenuation_mask, angles)

    return attenuated_volume


# @jit
# def generate_p0(N, angles, mu, dx, margin, shift,  dims=3):
#     """
#     Generate the initial pressure distribution p0 for a given tissue volume N and illumination angles.

#     Parameters
#     ----------
#     N : tuple
#         Size of the domain (2D or 3D).
#     angles : array
#         Array of illumination angles (azimuth and elevation) in degrees.
#     mu : float
#         The attenuation coefficient.
#     dx : float
#         The voxel size.
#     margin : ndarray
#         Margin for each side of corresponding dimension.
#     shift : ndarray
#         Sensor margin for each dimension.
#     dims : int
#         Number of dimensions (2 or 3).

#     Returns
#     -------
#     ndarray
#         The initial pressure distribution p0 with margins added.
#     """
#     # Illuminate vessels with given angles
#     attenuated_volume = attenuation_mask_directional_3d(angles, N-margin, dx, mu)

#     # Add margin for the PML
#         p0 = add_margin(
#                 attenuated_volume,
#                 N,
#                 margin // 2,
#                 shift=(0, 0, -1 * shift[2]),
#             )

#     if dims == 2:
#         # flatten the images
#         p0 = jnp.sum(jnp.array(p0), axis=1)


#     return p0


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

    # ----------------------
    # Generate vessels
    # ----------------------
    print("Generating vessels")

    tissue_margin = 2 * (np.array([u.PML_MARGIN] * 3) + np.array([u.PML_MARGIN] * 3))
    tissue_volume = np.array(u.N) - tissue_margin
    print(f"Tissue volume: {tissue_volume}")

    vessels_batch, n_iters = generate_vessels_3d(
        tissue_volume, u.BATCH_SIZE, shrink_factor=u.SHRINK_FACTOR
    )

    if u.DIMS == 2:
        vessels_batch = np.sum(vessels_batch, axis=1)

    # Save vessels
    # ----------------------
    os.makedirs(f"{u.DATA_PATH}LNet/", exist_ok=True)
    folder_index = (
        max(
            [
                int(filename.split("_")[0])
                for filename in os.listdir(f"{u.DATA_PATH}LNet/")
                if filename.split("_")[0].isdigit()
            ],
            default=-1,
        )
        + 1
    )

    for i, vessels in enumerate(vessels_batch):
        filename = f"{i+folder_index}_{n_iters[i]}"

        lnet_file = f"{u.DATA_PATH}LNet/{filename}.npy"
        np.save(lnet_file, vessels)
        print(f"Created vessels image {lnet_file}")

    # ----------------------
    # ----------------------
    # Simulation setup
    # ----------------------
    if u.DIMS == 2:
        N = tuple(u.N[:2])
        DX = tuple(u.DX[:2])
        domain = Domain(N, DX)

        # Set up the sensors
        x_start = u.SENSOR_MARGIN[1] + u.PML_MARGIN
        x_end = u.N[1] - u.SENSOR_MARGIN[1] - u.PML_MARGIN
        x = np.linspace(x_start, x_end, u.NUM_SENSORS)
        y = np.ones_like(x) * (u.SENSOR_MARGIN[0] + u.PML_MARGIN)
        sensor_positions = np.array([x, y])
        sensors_obj = BLISensors(sensor_positions, domain.N)

        attenuation_mask_directional_2d_vmap = vmap(
            attenuation_mask_directional_2d, in_axes=(0, None, None, None)
        )

    elif u.DIMS == 3:
        domain = Domain(u.N, u.DX)

        # Set up the sensors
        margin = np.array(u.SENSOR_MARGIN) + np.array([u.PML_MARGIN] * 3)
        sensor_positions = point_plane(u.NUM_SENSORS, u.N, margin)
        sensors_obj = BLISensors(sensor_positions, u.N)

    @u.timer
    @jit
    def compiled_simulator(medium, time_axis, p0):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    batch_compiled_simulator = vmap(compiled_simulator, in_axes=(None, None, 0))

    os.makedirs(f"{u.DATA_PATH}angles/", exist_ok=True)
    os.makedirs(f"{u.DATA_PATH}sensors/", exist_ok=True)
    os.makedirs(f"{u.DATA_PATH}p0/", exist_ok=True)
    os.makedirs(f"{u.DATA_PATH}p_data/", exist_ok=True)
    os.makedirs(f"{u.DATA_PATH}c/", exist_ok=True)

    # Loop through the LNet files which don't have a corresponding p0 file
    # ----------------------
    for file in os.listdir(f"{u.DATA_PATH}LNet/"):
        if exit_flag:
            break

        if os.path.exists(u.DATA_PATH + f"p0/{file.split('_')[0]}*"):
            continue
        file_index = file.split("_")[0]

        vessels = jnp.load(f"{u.DATA_PATH}LNet/{file}")

        if u.DIMS == 2:
            # Illumination
            # ----------------------
            angles = np.random.uniform(0, 360, u.NUM_LIGHTING_ANGLES)
            attenuated_vessels = attenuation_mask_directional_2d_vmap(
                angles, vessels, DX[0], u.MU
            )

            # Add margin to the vessels
            # ----------------------
            p0 = vmap(add_margin, in_axes=(0, None, None, None))(
                attenuated_vessels,
                N,
                tissue_margin // 2,
                (0, -1 * u.SENSOR_MARGIN[1]),
            )

            # Sound speed
            # ----------------------
            if u.SOUND_SPEED_VARIATION_AMPLITUDE == 0:
                sound_speed = u.C * jnp.ones(N)
            else:
                # sound_speed_area = np.array(N) - np.array(u.PML_MARGIN)
                noise = generate_perlin_noise_2d(
                    N,
                    # sound_speed_area,
                    [u.SOUND_SPEED_PERIODICITY] * 2,
                    tileable=(False, False),
                )
                sound_speed = u.C + u.SOUND_SPEED_VARIATION_AMPLITUDE * noise

        elif u.DIMS == 3:
            # Illumination
            # ----------------------
            azimuth_degs = np.random.uniform(0, 360, u.NUM_LIGHTING_ANGLES)
            elevation_degs = np.random.uniform(0, 180, u.NUM_LIGHTING_ANGLES)
            angles = jnp.stack([azimuth_degs, elevation_degs], axis=1)

            attenuated_vessels = batch_attenuate_light(vessels, u.MU, u.DX[0], angles)

            # Add margin to the vessels
            # ----------------------
            p0 = []
            for i in range(attenuated_vessels.shape[0]):
                p0.append(
                    device_put(
                        add_margin(
                            attenuated_vessels[i],
                            u.N,
                            tissue_margin // 2,
                            shift=(0, 0, -1 * u.SENSOR_MARGIN[2]),
                        ),
                        devices("cpu")[0],
                    )
                )
            # Sound speed
            # ----------------------
            if u.SOUND_SPEED_VARIATION_AMPLITUDE == 0:
                sound_speed = u.C * jnp.ones(u.N)
            else:
                # sound_speed_volume = np.array(N) - np.array(u.PML_MARGIN)
                noise = generate_perlin_noise_3d(
                    u.N,
                    # sound_speed_volume,
                    [u.SOUND_SPEED_PERIODICITY] * 3,
                    tileable=(False, False, False),
                )
                sound_speed = u.C + u.SOUND_SPEED_VARIATION_AMPLITUDE * noise

            p0 = device_put(p0, devices("cuda")[0])

        # Save the illumination angles
        # ----------------------
        angles_file = f"{u.DATA_PATH}angles/{file_index}.npy"
        jnp.save(angles_file, angles)

        # Save p0
        # ----------------------
        p0_file = f"{u.DATA_PATH}p0/{file_index}.npy"
        jnp.save(p0_file, p0)

        # Save sound speed
        # ----------------------
        c_file = f"{u.DATA_PATH}c/{file_index}.npy"
        jnp.save(c_file, sound_speed)

        # ----------------------

        # ----------------------
        # Perform simulation
        # ----------------------
        print(f"Simulating {file}")
        # print(sound_speed.shape, sound_speed.max(), sound_speed.min())
        sound_speed = FourierSeries(jnp.expand_dims(sound_speed, -1), domain)
        medium = Medium(domain=domain, sound_speed=jnp.ones(N) * u.C, pml_size=u.PML_MARGIN)
        time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)
        medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=u.PML_MARGIN)
        # print(time_axis.dt, time_axis.Nt)
        p0 = vmap(FourierSeries, (0, None))(p0, domain)

        p_data = batch_compiled_simulator(medium, time_axis, p0)

        # ----------------------

        # ----------------------
        # Save the data
        # # ----------------------
        sensors_file = f"{u.DATA_PATH}sensors/{file_index}.npy"
        jnp.save(sensors_file, sensor_positions)
        p_data_file = f"{u.DATA_PATH}p_data/{file_index}.npy"
        jnp.save(p_data_file, p_data)

        print(f"Saved {p0_file}, {p_data_file}, {sensors_file}, {c_file}")
