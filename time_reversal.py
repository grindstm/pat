import yaml
from functools import partial

import jax
import jax.numpy as jnp
from jax.numpy.fft import fft2, fftfreq, fftshift
from jax import grad, jit, value_and_grad, vmap, nn
from jax.lax import scan
from jax import random
from jaxdf.operators import compose

from jwave import FourierSeries
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation
from jwave.signal_processing import smooth

import optax
from optax import adam
import numpy as np
import util as u
import generate_data as gd
from jaxdf.operators.differential import laplacian
# from generate_data import attenuation_mask_directional_2d, pad_0_wrapper, pad_1_wrapper

import matplotlib.pyplot as plt

from tqdm import tqdm

if u.DIMS == 2:
    N = u.N[:2]
    DX = u.DX[:2]


# _________________________________________________________________
def multi_illumination_parallel_optimized(
    P_data, sensor_positions, ATT_masks, num_iterations=10, learning_rate=1
):
    domain = Domain(N, DX)
    medium = Medium(domain=domain, sound_speed=jnp.ones(N) * u.C, pml_size=u.PML_MARGIN[0])
    time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)

    ATT_masks = vmap(gd.pad_1_wrapper, in_axes=(0,None))(ATT_masks, u.TISSUE_MARGIN[:2])
    ATT_masks = FourierSeries(jnp.expand_dims(ATT_masks, -1), domain)

    sensors_obj = BLISensors(positions=np.array(sensor_positions), n=domain.N)

    def simulate(medium, time_axis, p0):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    batch_compiled_simulate = vmap(simulate, in_axes=(None, None, 0))

    @jit
    def mse_loss(mu, c, ATT_masks, P_data):
        P_0 = mu * ATT_masks
        medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN[0])
        P_pred = batch_compiled_simulate(medium, time_axis, P_0)
        return 0.5 * jnp.sum(jnp.abs(P_pred.squeeze() - P_data) ** 2)
    
    @jit
    def mse_loss_l2mu_smoothc(mu, c, ATT_masks, P_data):
        
        P_0 = mu * ATT_masks
        
        medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN[0])
        
        P_pred = batch_compiled_simulate(medium, time_axis, P_0)
        
        R_mu = 0.5 * jnp.sum(jnp.abs(mu.on_grid)**2)

        # ---------------------
        R_c = 0.
        
        # R_c = 0.5 * jnp.sum(jnp.abs(laplacian(c).on_grid)**2)

        # c_og = c.on_grid / jnp.mean(c.on_grid)
        # shape = c_og.shape
        # fft_c = fft2(c_og)
        # fft_c_shifted = fftshift(fft_c)  
        # freq_x = fftfreq(shape[0]).reshape(-1, 1)
        # freq_y = fftfreq(shape[1]).reshape(1, -1)
        # freq_x_shifted = fftshift(freq_x)
        # freq_y_shifted = fftshift(freq_y)
        # k = 0.5 * jnp.sqrt(freq_x_shifted ** 2 + freq_y_shifted ** 2)
        # epsilon = 1e-10
        # k_safe = 50* k + epsilon 
        # g_k = jnp.maximum(jnp.ones_like(k)*-3, 100 * jnp.log(k_safe))
        # R_c = 0.5 * jnp.sum(g_k * jnp.abs(fft_c)**2)
        # ---------------------
        
        fidelity = 0.5 * jnp.sum(jnp.abs(P_pred.squeeze() - P_data) ** 2)
        return fidelity + R_mu + R_c

    mu = FourierSeries(jnp.zeros(N), domain)
    c = FourierSeries(jnp.ones(N) * u.C, domain)

    mses = []
    p_rs = []
    c_rs = []

    opt = adam(learning_rate)
    params = (mu, c)
    opt_state = opt.init(params)

    for i in tqdm(range(num_iterations)):
        # print(i)
        loss, gradients = value_and_grad(mse_loss_l2mu_smoothc, argnums=(0, 1))(
            mu, c, ATT_masks, P_data
        )
        # loss, gradients = value_and_grad(mse_loss, argnums=(0, 1))(
        #     mu, c, ATT_masks, P_data
        # )

        updates, opt_state = opt.update(gradients, opt_state)
        new_params = optax.apply_updates(params, updates)
        mu_params = jnp.clip(new_params[0].params, 0)
        mu = mu.replace_params(mu_params)
        c = new_params[1]
        params = (mu, c)

        mses.append(loss)
        p_rs.append(mu.on_grid)
        c_rs.append(c.on_grid)

    return p_rs, c_rs, mses


# --------------------------------------------


def multi_illumination(
    p_data, sensor_positions, angles, num_iterations=10, learning_rate=1
):
    domain = Domain(N, DX)
    medium = Medium(domain=domain, sound_speed=jnp.ones(N) * u.C, pml_size=u.PML_MARGIN)
    time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)

    vessels_shape = tuple(N - np.array(2 * [2 * u.PML_MARGIN]))

    pml_mask = add_margin(
        jnp.ones(vessels_shape), N, (u.PML_MARGIN, u.PML_MARGIN), (0, 0)
    )
    pml_mask = FourierSeries(jnp.expand_dims(pml_mask, -1), domain)

    sensors_obj = BLISensors(positions=np.array(sensor_positions), n=domain.N)

    def simulate(medium, time_axis, p0):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    @jit
    def mse_loss(p0, c, attenuation_mask, p_data):
        p0 = p0.replace_params(p0.params * attenuation_mask.params)
        c = c.replace_params(c.params)
        medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN)
        p_pred = simulate(medium, time_axis, p0)
        return 0.5 * jnp.sum(jnp.abs(p_pred - p_data) ** 2)

    def update(p0, c):
        for i, angle in enumerate(angles):
            attenuation_mask = attenuation_mask_directional_2d(
                angle, jnp.ones(vessels_shape), DX[0], u.ATTENUATION
            )
            attenuation_mask = add_margin(
                attenuation_mask, N, (u.PML_MARGIN, u.PML_MARGIN), (0, 0)
            )
            attenuation_mask = FourierSeries(
                jnp.expand_dims(attenuation_mask, -1), domain
            )

            loss, gradients = value_and_grad(mse_loss, argnums=(0, 1))(
                p0, c, attenuation_mask, p_data[i]
            )
            new_p0 = p0 - learning_rate * gradients[0]
            new_c = c - 1000 * learning_rate * gradients[1]
            mses.append(loss)
            p_rs.append(new_p0.on_grid)
            c_rs.append(new_c.on_grid)
        return new_p0, new_c, gradients

    p0 = FourierSeries(jnp.expand_dims(jnp.zeros(N), -1), domain)
    c = FourierSeries(jnp.expand_dims(jnp.ones(N) * u.C, -1), domain)

    mses = []
    p_rs = []
    c_rs = []
    for i in range(num_iterations):
        print(i)
        p0, c, gradients = update(p0, c)
        jnp.save(f"{u.DATA_PATH}/gradient_{i}.npy", gradients[1].on_grid[..., 0])

    return p_rs, c_rs, mses


# --------------------------------------------


def multi_illumination_parallel(
    p_data, sensor_positions, angles, num_iterations=10, learning_rate=1
):
    domain = Domain(N, DX)
    medium = Medium(domain=domain, sound_speed=jnp.ones(N) * u.C, pml_size=u.PML_MARGIN)
    time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)

    vessels_shape = tuple(N - np.array(2 * [2 * u.PML_MARGIN]))

    attenuation_masks = vmap(
        attenuation_mask_directional_2d, in_axes=(0, None, None, None)
    )(angles, jnp.ones(vessels_shape), DX[0], u.ATTENUATION)
    attenuation_masks = vmap(add_margin, in_axes=(0, None, None, None))(
        attenuation_masks, N, (u.PML_MARGIN, u.PML_MARGIN), (0, 0)
    )
    attenuation_masks = FourierSeries(jnp.expand_dims(attenuation_masks, -1), domain)

    pml_mask = add_margin(
        jnp.ones(vessels_shape), N, (u.PML_MARGIN, u.PML_MARGIN), (0, 0)
    )
    pml_mask = FourierSeries(jnp.expand_dims(pml_mask, -1), domain)

    sensors_obj = BLISensors(positions=np.array(sensor_positions), n=domain.N)

    # @jit
    def simulate(medium, time_axis, p0):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    batch_compiled_simulate = vmap(simulate, in_axes=(None, None, 0))

    @jit
    def mse_loss(p0, c, attenuation_mask, p_data):
        p0 = p0.replace_params(p0.params * attenuation_mask.params)
        c = c.replace_params(c.params)
        medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN)
        p_pred = batch_compiled_simulate(medium, time_axis, p0)
        return 0.5 * jnp.sum(jnp.abs(p_pred - p_data) ** 2)

    def update(p0, c):
        loss, gradients = value_and_grad(mse_loss, argnums=(0, 1))(
            p0, c, attenuation_masks, p_data
        )
        new_p0 = p0 - learning_rate * gradients[0]
        new_c = c - 1000 * learning_rate * gradients[1]
        mses.append(loss)
        p_rs.append(new_p0.on_grid)
        c_rs.append(new_c.on_grid)
        return new_p0, new_c, gradients

    p0 = FourierSeries(jnp.expand_dims(jnp.zeros(N), -1), domain)
    c = FourierSeries(jnp.expand_dims(jnp.ones(N) * u.C, -1), domain)

    mses = []
    p_rs = []
    c_rs = []
    for i in range(num_iterations):
        print(i)
        p0, c, gradients = update(p0, c)
        jnp.save(f"{u.DATA_PATH}/gradient_{i}.npy", gradients[1].on_grid[..., 0])

    return p_rs, c_rs, mses


# _________________________________________________________________


@jit
def lazy_time_reversal(p_data, sensor_positions):
    domain = Domain(N, DX)
    sound_speed = jnp.ones(N) * u.C
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=u.PML_MARGIN)
    time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)

    def simulate(p0, sensors_obj):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

    sensors_obj = BLISensors(positions=sensor_positions, n=u.N)

    def mse_loss(p0, p_data):
        p0 = p0.replace_params(p0.params)
        p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
        return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2)

    p0 = FourierSeries.empty(domain)

    loss, gradient = value_and_grad(mse_loss)(p0, p_data)

    return -gradient, loss


# @partial(jit, static_argnums=(3, 4))
# def iterative_time_reversal(
#     p0, p_data, sensor_positions, num_iterations=10, learning_rate=0.1
# ):
#     mses = jnp.zeros(num_iterations)

#     sensors_obj = BLISensors(positions=sensor_positions, n=u.N)

#     mask = jnp.ones_like(p0).at[..., N[2] - u.PML_MARGIN - u.SENSOR_MARGIN[2] :].set(0)
#     mask = FourierSeries(mask, domain)

#     def mse_loss(p0, p_data):
#         p0 = p0.replace_params(p0.params * mask.params)
#         p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
#         return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2)

#     p0 = FourierSeries.empty(domain)

#     def update(p0, i):
#         loss, gradients = value_and_grad(mse_loss)(p0, p_data)
#         new_p0 = p0 - learning_rate * gradients
#         return new_p0, (new_p0.on_grid, loss)

#     # Initialize p0 for scan and perform the iterations
#     _, (all_p0s, mses) = scan(update, p0, None, length=num_iterations)

#     return all_p0s, mses


# @partial(jit, static_argnums=(3, 4))
# def iterative_time_reversal(
#     p0, p_data, sensor_positions, num_iterations=10, learning_rate=0.1
# ):
#     sensors_obj = BLISensors(positions=sensor_positions, n=N)

#     # Mask to zero regions outside the tissue
#     mask = jnp.ones_like(p0).at[..., N[2] - PML_MARGIN - SENSOR_MARGIN[2] :].set(0)
#     mask = FourierSeries(mask, domain)

#     def mse_loss(p0, p_data):
#         # Apply mask to p0 parameters
#         p0 = p0.replace_params(p0.params * mask.params)
#         p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
#         return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2)

#     p0 = FourierSeries.empty(domain)

#     def update(p0, i):
#         loss, gradients = value_and_grad(mse_loss)(p0, p_data)
#         new_p0 = p0 - learning_rate * gradients
#         return new_p0, (new_p0.on_grid, loss)

#     # Initialize p0 for scan and perform the iterations
#     _, (all_p0s, mses) = scan(update, p0, None, length=num_iterations)

#     return all_p0s, mses


# @partial(jit, static_argnums=(3))
# def iterative_time_reversal_optimized(p0, p_data, sensor_positions, num_iterations=10):
#     sensors_obj = BLISensors(positions=sensor_positions, n=u.N)

#     # Define the loss function
#     def mse_loss(p0):
#         p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
#         return 0.5 * jnp.sum((p_pred - p_data) ** 2)

#     # Initialize the optimizer
#     optimizer = optax.adam(learning_rate=0.1)
#     opt_state = optimizer.init(p0)

#     # Update function to be used with scan
#     def update(opt_state, _):
#         loss, grads = value_and_grad(mse_loss)(p0)
#         updates, new_opt_state = optimizer.update(grads, opt_state)
#         new_p0 = optax.apply_updates(p0, updates)
#         return new_opt_state, (new_p0, loss)

#     # Perform optimization
#     final_opt_state, (p0, losses) = scan(update, opt_state, None, length=num_iterations)

#     return p0, losses
