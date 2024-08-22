import os
import signal
import argparse
import time

import numpy as np
from v_system.VSystemGenerator import VSystemGenerator
import scipy.ndimage
from perlin_numpy import generate_perlin_noise_3d, generate_perlin_noise_2d

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation
from jwave.signal_processing import smooth

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax import random


# __ CPU
from jax import device_put, devices

# __ map
import jax.lax as lax
from functools import partial

import util as u

jax.clear_caches()

# ______________________________________________________


def generate_vessels_3d(N, shrink_factor=1):
    """
    Generate a 3D volume of vessels.

    Parameters
    ----------
    N : tuple
        The size of the domain
    shrink_factor : float
        The factor by which to grow the domain for the generation of the vessels. The domain is then shrunk back to the original size. Use this if generated vessels contain artifacts or are too fat/simple.
    """
    sim = VSystemGenerator(
        tissue_volume=np.array(N) * shrink_factor, d0_mean=20.0, d0_std=5.0
    )
    mu, n_iter = sim.create_network()

    if shrink_factor != 1:
        mu = scipy.ndimage.zoom(mu, 1 / shrink_factor)

    return mu, n_iter


def generate_mu_2d(mu_3d):
    return jnp.sum(mu_3d, axis=0)


# ______________________________________________________


def line_sensor(N, margin, num_sensors):
    x = np.linspace(margin[0], N[0] - margin[0], num_sensors)
    y = np.ones_like(x) * (N[1] - margin[1])
    return np.array([x, y])


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


# ______________________________________________________


@jit
def attenuation_mask_directional_2d(angle, volume, dx, attenuation, r):
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
    attenuation : float
        The attenuation coefficient.
    r : int
        The radius of the volume.

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
    X = X - width // 2 + r * ux
    Y = Y - height // 2 + r * uy

    distances = ux * X * dx + uy * Y * dx
    mask = jnp.exp(-attenuation * distances)
    result = jnp.clip(mask * volume, max=1.0)
    return result


attenuation_mask_directional_2d_vmap = vmap(
    attenuation_mask_directional_2d, in_axes=(0, None, None, None, None)
)


@jit
def attenuation_mask_directional_3d(angles, volume, dx, attenuation, r):
    """
    Compute the attenuation mask for a 3D volume given an angle, voxel size, and attenuation coefficient.

    Parameters
    ----------
    angles : ndarray
        The direction angles in degrees.
    volume : ndarray
        The 3D volume to be attenuated.
    dx : float
        The voxel size.
    attenuation : float
        The attenuation coefficient.
    r : int
        The radius of the volume.

    Returns
    -------
    ndarray
        The attenuated 3D volume.
    """
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
    X = X - width // 2 + r * ux
    Y = Y - height // 2 + r * uy
    Z = Z - depth // 2 + r * uz

    distances = ux * X * dx + uy * Y * dx + uz * Z * dx
    mask = jnp.exp(-attenuation * distances)
    result = jnp.clip(mask * volume, max=1.0)
    return result


@jit
def batch_attenuate_light(volume, attenuation, dx, angles):
    """
    Returns a batch of images for a volume illuminated at given angles. Use this to avoid the overhead of vmap when generating 3D data.

    Parameters
    ----------
    volume : ndarray
        The voxel volume of the tissue
    attenuation : float
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
        attenuation_mask_directional_3d, volume=volume, dx=dx, attenuation=attenuation
    )
    attenuated_volume = lax.map(partial_attenuation_mask, angles)

    return attenuated_volume


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

pad_0_wrapper = partial(jnp.pad, mode="constant", constant_values=0)
pad_1_wrapper = partial(jnp.pad, mode="constant", constant_values=1)


def generate_2d_data(mu):
    """
    Generate 2D data

    Parameters
    ----------
    mu : ndarray
        The 3D volume of the tissue generated using generate_vessels_3d

    Returns
    -------
    tuple
        The initial pressure, attenuation masks, sound speed, and simulated data
    """

    N = u.N[:2]
    DX = u.DX[:2]
    TISSUE_MARGIN = u.TISSUE_MARGIN[:2]

    mu = generate_mu_2d(mu)  # Sum the 3D volume to get the 2D volume

    # Illumination
    # ----------------------
    angles = jnp.linspace(0, 360, u.NUM_LIGHTING_ANGLES, endpoint=False)
    jnp.save(u.file(u.angles_path, i), angles)

    ATT_masks = attenuation_mask_directional_2d_vmap(
        angles, jnp.ones(N), DX[0], u.ATTENUATION, N[0] / 2 - u.TISSUE_MARGIN[0]
    )

    mu = pad_0_wrapper(mu, TISSUE_MARGIN)

    P_0 = ATT_masks * mu

    # Sound speed
    # ----------------------
    if u.C_VARIATION_AMPLITUDE == 0:
        c = u.C * jnp.ones(N)
    else:
        noise = generate_perlin_noise_2d(
            N,
            [u.C_PERIODICITY] * 2,
            tileable=(False, False),
        )
        mu_binary = jnp.where(mu > 0, 1, 0)
        c = u.C + u.C_VARIATION_AMPLITUDE * jnp.array(noise)
        c = c.at[mu_binary.astype(bool)].set(u.C_BLOOD)
    # ----------------------

    # ----------------------
    # Perform simulation
    # ----------------------
    print(f"Simulating")
    c = FourierSeries(jnp.expand_dims(c, -1), domain)
    medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN[0])
    P_0 = vmap(FourierSeries, (0, None))(P_0, domain)
    P_data = batch_compiled_simulator(medium, time_axis, P_0)[0]

    return P_0, ATT_masks, c, P_data
    # ----------------------


def generate_3d_data(mu):
    """
    Generate 3D data

    Parameters
    ----------
    mu : ndarray
        The 3D volume of the tissue generated using generate_vessels_3d

    Returns
    -------
    tuple
        The initial pressure, attenuation masks, sound speed, and simulated data
    """

    # Illumination
    # ----------------------
    azimuth_degs = np.random.uniform(0, 360, u.NUM_LIGHTING_ANGLES)
    elevation_degs = np.random.uniform(0, 180, u.NUM_LIGHTING_ANGLES)
    angles = jnp.stack([azimuth_degs, elevation_degs], axis=1)

    ATT_masks = batch_attenuate_light(jnp.ones_like(mu), u.ATTENUATION, DX[0], angles)
    mu = pad_0_wrapper(mu, TISSUE_MARGIN)

    # Add margin to the vessels
    # ----------------------
    P_0 = ATT_masks * mu

    # Sound speed
    # ----------------------
    if u.C_VARIATION_AMPLITUDE == 0:
        c = u.C * jnp.ones(N)
    else:
        noise = generate_perlin_noise_3d(
            N,
            [u.C_PERIODICITY] * 3,
            tileable=(False, False, False),
        )
        mu_binary = jnp.where(mu > 0, 1, 0)
        c = u.C + u.C_VARIATION_AMPLITUDE * jnp.array(noise)
        c = c.at[mu_binary.astype(bool)].set(u.C_BLOOD)
    # ----------------------

    # ----------------------
    # Perform simulation
    # ----------------------
    print(f"Simulating")
    c = FourierSeries(jnp.expand_dims(c, -1), domain)
    medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN[0])
    P_0 = vmap(FourierSeries, (0, None))(P_0, domain)
    P_data = batch_compiled_simulator(medium, time_axis, P_0)

    return P_0, ATT_masks, c, P_data


# ----------------------


if __name__ == "__main__":
    N = u.N
    DX = u.DX
    SENSOR_MARGIN = u.SENSOR_MARGIN
    PML_MARGIN = u.PML_MARGIN
    TISSUE_MARGIN = u.TISSUE_MARGIN

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
    # ----------------------

    # ----------------------
    # Simulation setup
    # ----------------------
    if u.DIMS == 2:
        domain = Domain(N[:2], DX[:2])

        # Set up the sensors
        sensor_positions = line_sensor(N[:2], u.SENSOR_MARGIN[:2], u.NUM_SENSORS)
        sensors_obj = BLISensors(sensor_positions, domain.N)

        medium = Medium(
            domain=domain, sound_speed=jnp.ones(N[:2]) * u.C, pml_size=u.PML_MARGIN[0]
        )
        time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)

    elif u.DIMS == 3:
        domain = Domain(N, DX)

        # Set up the sensors
        sensor_positions = point_plane(N, u.SENSOR_MARGIN, u.NUM_SENSORS)
        sensors_obj = BLISensors(sensor_positions, domain.N)
        medium = Medium(
            domain=domain, sound_speed=jnp.ones(N) * u.C, pml_size=u.PML_MARGIN[0]
        )
        time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)

    @u.timer
    @jit
    def compiled_simulator(medium, time_axis, p0):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    batch_compiled_simulator = vmap(compiled_simulator, in_axes=(None, None, 0))

    # ----------------------
    for i in range(u.BATCH_SIZE):
        if exit_flag:
            break

        # ----------------------
        # Generate mu
        # ----------------------
        print("Generating mu")
        tissue_volume = np.array(N) - 2 * np.array(u.TISSUE_MARGIN)
        print(f"Tissue volume: {tissue_volume}")
        mu, n_iter = generate_vessels_3d(tissue_volume, u.SHRINK_FACTOR)

        if u.DIMS == 2:
            P_0, ATT_masks, c, P_data = generate_2d_data(mu)
            jnp.save(u.file(u.mu_path, i), mu)

        elif u.DIMS == 3:
            P_0, ATT_masks, c, P_data = generate_3d_data(mu)

        # ----------------------
        # Add noise to the data
        # ----------------------
        key = random.PRNGKey(int(time.time()))
        keys = random.split(key, u.NUM_LIGHTING_ANGLES)
        P_data_noisy = add_colored_noise_vmap(keys, P_data, 1.0, u.NOISE_AMPLITUDE)

        # ----------------------
        # Save the data
        # ----------------------
        jnp.save(u.file(u.P_0_path, i), P_0.on_grid.squeeze())
        jnp.save(u.file(u.ATT_masks_path, i), ATT_masks)
        jnp.save(u.file(u.mu_path, i), mu)
        jnp.save(u.file(u.c_path, i), c.on_grid.squeeze())
        jnp.save(u.file(u.sensors_path, i), sensor_positions)
        jnp.save(u.file(u.P_data_path, i), P_data.squeeze())
        jnp.save(u.file(u.P_data_noisy_path, i), P_data_noisy.squeeze())

        print(f"Saved {i}")
