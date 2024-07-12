import yaml
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap, nn
from jax.lax import scan
from jax import random
from jaxdf.operators import compose

from jwave import FourierSeries
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation
from jwave.signal_processing import smooth

from optax import adam
import numpy as np
import util as u
from generate_data import attenuation_mask_directional_2d, add_margin

# Set up the simulator
# ----------------------
if u.DIMS == 2:
    N = u.N[:2]
    DX = u.DX[:2]

# domain = Domain(N, DX)
# sound_speed = jnp.ones(N) * u.C
# medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=u.PML_MARGIN)
# time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)


# def simulate(p0, sensors_obj):
#     return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)
import matplotlib.pyplot as plt

# _________________________________________________________________


def multi_illumination(
    p_data, sensor_positions, angles, num_iterations=10, learning_rate=1
):
    domain = Domain(N, DX)
    medium = Medium(domain=domain, sound_speed=jnp.ones(N) * u.C, pml_size=u.PML_MARGIN)
    time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)

    vessels_shape = tuple(N - np.array(2 * [2 * u.PML_MARGIN]))

    # attenuation_masks = vmap(
    #     attenuation_mask_directional_2d, in_axes=(0, None, None, None)
    # )(angles, jnp.ones(vessels_shape), DX[0], u.MU)
    # attenuation_masks = vmap(add_margin, in_axes=(0, None, None, None))(
    #     attenuation_masks, N, (u.PML_MARGIN, u.PML_MARGIN), (0, 0)
    # )
    # attenuation_masks = FourierSeries(jnp.expand_dims(attenuation_masks, -1), domain)

    pml_mask = add_margin(
        jnp.ones(vessels_shape), N, (u.PML_MARGIN, u.PML_MARGIN), (0, 0)
    )
    pml_mask = FourierSeries(jnp.expand_dims(pml_mask, -1), domain)

    sensors_obj = BLISensors(positions=np.array(sensor_positions), n=domain.N)

    # @jit
    def simulate(medium, time_axis, p0):
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)


    def mse_loss(p0, c, attenuation_mask, p_data=p_data):
        p0 = p0.replace_params(p0.params * attenuation_mask.params)
        c = u.C - 10 + 100.0 * compose(c.params)(nn.sigmoid) * pml_mask
        medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN)
        p_pred = simulate(medium, time_axis, p0)
        return 0.5 * jnp.sum(jnp.abs(p_pred - p_data) ** 2)
    
    # batch_compiled_simulate = vmap(simulate, in_axes=(None, None, 0))
    
    # def mse_loss(p0, c, p_data=p_data):
    #     p0 = p0.replace_params(p0.params * attenuation_masks.params)
    #     c = u.C - 10 + 100.0 * compose(c.params)(nn.sigmoid) * pml_mask
    #     medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN)
    #     p_pred = batch_compiled_simulate(medium, time_axis, p0)
    #     return 0.5 * jnp.sum(jnp.abs(p_pred - p_data) ** 2)

    def update(p0, c):
        for angle in angles:
            attenuation_mask = attenuation_mask_directional_2d(angle, jnp.ones(vessels_shape), DX[0], u.MU)

            attenuation_mask =add_margin(attenuation_mask, N, (u.PML_MARGIN, u.PML_MARGIN), (0, 0))
            attenuation_mask = FourierSeries(jnp.expand_dims(attenuation_mask, -1), domain)

            loss, gradients = value_and_grad(mse_loss, argnums=(0, 1))(p0, c, attenuation_mask)
            new_p0 = p0 - learning_rate * gradients[0]
            new_c = c - learning_rate * gradients[1]
            mses.append(loss)
            p_rs.append(new_p0.on_grid)
            c_rs.append(new_c.on_grid)
        return new_p0, new_c

    # p0 = jnp.empty([u.NUM_LIGHTING_ANGLES, *N])
    # p0 = vmap(FourierSeries, (0, None))(p0, domain)

    p0 = FourierSeries(jnp.expand_dims(jnp.zeros(N), -1), domain)
    c = FourierSeries(jnp.expand_dims(jnp.ones(N) * u.C, -1), domain)

    mses = []
    p_rs = []
    c_rs = []
    for i in range(num_iterations):
        print(i)
        p0, c = update(p0, c)

    return p_rs, c_rs, mses


# _________________________________________________________________


@jit
def lazy_time_reversal(p_data, sensor_positions):

    sensors_obj = BLISensors(positions=sensor_positions, n=u.N)

    def mse_loss(p0, p_data):
        p0 = p0.replace_params(p0.params)
        p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
        return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2)

    p0 = FourierSeries.empty(domain)

    loss, gradient = value_and_grad(mse_loss)(p0, p_data)

    return -gradient, loss


@partial(jit, static_argnums=(3, 4))
def iterative_time_reversal(
    p0, p_data, sensor_positions, num_iterations=10, learning_rate=0.1
):
    mses = jnp.zeros(num_iterations)

    sensors_obj = BLISensors(positions=sensor_positions, n=u.N)

    mask = jnp.ones_like(p0).at[..., N[2] - u.PML_MARGIN - u.SENSOR_MARGIN[2] :].set(0)
    mask = FourierSeries(mask, domain)

    def mse_loss(p0, p_data):
        p0 = p0.replace_params(p0.params * mask.params)
        p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
        return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2)

    p0 = FourierSeries.empty(domain)

    def update(p0, i):
        loss, gradients = value_and_grad(mse_loss)(p0, p_data)
        new_p0 = p0 - learning_rate * gradients
        return new_p0, (new_p0.on_grid, loss)

    # Initialize p0 for scan and perform the iterations
    _, (all_p0s, mses) = scan(update, p0, None, length=num_iterations)

    return all_p0s, mses


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


@partial(jit, static_argnums=(3))
def iterative_time_reversal_optimized(p0, p_data, sensor_positions, num_iterations=10):
    sensors_obj = BLISensors(positions=sensor_positions, n=u.N)

    # Define the loss function
    def mse_loss(p0):
        p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
        return 0.5 * jnp.sum((p_pred - p_data) ** 2)

    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(p0)

    # Update function to be used with scan
    def update(opt_state, _):
        loss, grads = value_and_grad(mse_loss)(p0)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_p0 = optax.apply_updates(p0, updates)
        return new_opt_state, (new_p0, loss)

    # Perform optimization
    final_opt_state, (p0, losses) = scan(update, opt_state, None, length=num_iterations)

    return p0, losses
