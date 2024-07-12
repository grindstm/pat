from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation

from jax import grad, jit
import jax.numpy as jnp
import numpy as np

from generate_data import illuminate_vessels, add_margin
import util as u

tissue_margin = 2 * (np.array([u.PML_MARGIN] * 3) + np.array([u.PML_MARGIN] * 3))


def batch_generate_p0(N, angles):

    return jnp.sum(p0, axis=0)


@jit
def compiled_simulator(medium, time_axis, p0, sensors_obj):
    return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)


batched_compiled_simulator = jax.vmap(compiled_simulator, in_axes=(None, None, 0, None))


def forward_operator(p0, c, angles, domain, time_axis, sensors):
    # batch simulate the domain
    generated_p0 = generate_p0(p0, angles)
    sound_speed = FourierSeries(jnp.expand_dims(sound_speed, -1), domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=u.PML_MARGIN)
    time_axis = TimeAxis.from_medium(medium, cfl=CFL)
    sensors_obj = BLISensors(sensors, domain.N)
    p0 = FourierSeries(p0, domain)
    return compiled_simulator(medium, time_axis, generated_p0, sensors_obj)


# Define the objective function
def objective_function(
    p0,
    c,
    p_data,
    azimuth_degs,
    elevation_degs,
    domain,
    time_axis,
    sensors,
    lambda1,
    lambda2,
):
    num_illuminations = azimuth_degs.shape[0]
    simulated_data = []

    for i in range(num_illuminations):
        sim_data = forward_operator(
            p0, c, azimuth_degs[i], elevation_degs[i], domain, time_axis, sensors
        )
        simulated_data.append(sim_data)

    simulated_data = jnp.stack(simulated_data)

    data_fidelity = 0.5 * jnp.sum((simulated_data - p_data) ** 2)
    regularization = lambda1 * jnp.sum(p0**2) + lambda2 * jnp.sum(jnp.gradient(c) ** 2)

    return data_fidelity + regularization


# Initialize parameters
p0_init = jnp.zeros_like(p_data[0])
c_init = jnp.ones_like(p_data[0])

# Define the gradients
gradients = grad(objective_function, argnums=(0, 1))

# Set up the optimizer
opt_init, opt_update, get_params = adam(step_size=1e-3)
opt_state = opt_init((p0_init, c_init))

sensors = jnp.load("sensors/{0}.npy")


@jit
def step(
    i,
    opt_state,
    p_data,
    azimuth_degs,
    elevation_degs,
    domain,
    time_axis,
    sensors,
    lambda1,
    lambda2,
):
    p0, c = get_params(opt_state)
    grads = gradients(
        p0,
        c,
        p_data,
        azimuth_degs,
        elevation_degs,
        domain,
        time_axis,
        sensors,
        lambda1,
        lambda2,
    )
    return opt_update(i, grads, opt_state)


# Perform optimization
num_steps = 4
for i in range(num_steps):
    opt_state = step(
        i,
        opt_state,
        p_data,
        azimuth_degs,
        elevation_degs,
        domain,
        time_axis,
        sensors,
        lambda1,
        lambda2,
    )
    if i % 100 == 0:
        p0, c = get_params(opt_state)
        loss = objective_function(
            p0,
            c,
            p_data,
            azimuth_degs,
            elevation_degs,
            domain,
            time_axis,
            sensors,
            lambda1,
            lambda2,
        )
        print(f"Step {i}, Loss: {loss}")
