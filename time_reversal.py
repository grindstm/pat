import yaml
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from jax.lax import scan
from jax import random

from jwave import FourierSeries
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation
from jwave.signal_processing import smooth

import optax

import util


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
# Set up the simulator
domain = Domain(N, DX)
sound_speed = jnp.ones(N) * C
medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=PML_MARGIN)
time_axis = TimeAxis.from_medium(medium, cfl=CFL)


def simulate(p0, sensors_obj):
    return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)


@jit
def lazy_time_reversal(p0, p_data, sensor_positions):

    sensors_obj = BLISensors(positions=sensor_positions, n=N)

    def mse_loss(p0, p_data):
        p0 = p0.replace_params(p0.params)
        p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
        return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2)

    p0 = FourierSeries.empty(domain)

    p_grad = grad(mse_loss)(p0, p_data)

    return -p_grad


@partial(jit, static_argnums=(3, 4))
def iterative_time_reversal(
    p0, p_data, sensor_positions, num_iterations=10, learning_rate=0.1
):

    mses = jnp.zeros(num_iterations)
    sensors_obj = BLISensors(positions=sensor_positions, n=N)

    mask = jnp.ones_like(p0).at[..., N[2]-PML_MARGIN-SENSOR_MARGIN[2]:].set(0)
    mask = FourierSeries(mask, domain)
    
    def mse_loss(p0, p_data):
        p0 = p0.replace_params(p0.params*mask.params)
        p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
        return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2)


    p0 = FourierSeries.empty(domain)

    # for i in range(num_iterations):
    #     loss, gradients = value_and_grad(mse_loss)(p0, p_data)
    #     mses.append(loss)
    #     # p0 = p0.replace_params(p0.params)
    #     p0 -= learning_rate * gradients

    def update(p0, i):
        loss, gradients = value_and_grad(mse_loss)(p0, p_data)
        p0 -= learning_rate * gradients

        # return p0, (p0, loss)  # Pass both updated p0 and loss
        return p0, loss

    # Use scan to perform the iterations
    p0, mses = scan(update, p0, jnp.arange(num_iterations))

    return p0, mses

@partial(jit, static_argnums=(3))
def iterative_time_reversal_optimized(p0, p_data, sensor_positions, num_iterations=10):
    sensors_obj = BLISensors(positions=sensor_positions, n=N)

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


#     @jit
#     def lazy_time_reversal(p0, p_data,sensors_obj):
#         def mse_loss(p0_params, p_data):
#             p0 = p0.replace_params(p0_params)
#             p_pred = simulate(p0, sensors_obj=sensors_obj)[..., 0]
#             regularization_strength = 0.01  # This is a hyperparameter you can tune
#             l2_norm = regularization_strength * jnp.sum(p0_params**2)
#             return 0.5 * jnp.sum(jnp.abs(p_pred - p_data[..., 0]) ** 2) + l2_norm

#         p0 = FourierSeries.empty(domain)

#         p_grad = grad(mse_loss)(p0, p_data)

#         return -p_grad


#     # @jit
#     # def iterative_time_reversal(p0, p_data, sensors_obj, num_iters=2, learning_rate=0.5):
#     #     def mse_loss(p0_params):
#     #         # Reconstruct p0 from parameters
#     #         p0_reconstructed = FourierSeries(p0_params, domain)
#     #         # Run simulation with current p0 estimate
#     #         p_pred = simulate(p0_reconstructed, sensors_obj=sensors_obj)[..., 0]
#     #         # Calculate mean squared error
#     #         return 0.5 * jnp.sum(jnp.square(p_pred - p_data[..., 0]))

#     #     # Initialize parameters (could also be passed as an argument)
#     #     p0_params = p0.params

#     #     # Define the gradient function
#     #     grad_loss = grad(mse_loss)

#     #     # Iterative update
#     #     for _ in range(num_iters):
#     #         # Compute gradients
#     #         gradients = grad_loss(p0_params)

#     #         # Update parameters using gradient descent
#     #         p0_params -= learning_rate * gradients

#     #     # Return the final reconstructed parameters
#     #     return FourierSeries(p0_params, domain)
