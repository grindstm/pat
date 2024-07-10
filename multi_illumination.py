from jax import grad, jit
from jax.experimental.optimizers import adam
import jax.numpy as jnp
import numpy as np

from generate_data import 

# Define the function to generate p0 based on illumination angles
def generate_p0(tissue_volume, azimuth_deg, elevation_deg):

    # return tissue_volume * (np.cos(np.deg2rad(azimuth_deg)) + np.sin(np.deg2rad(elevation_deg)))

# Define the forward operator
def forward_operator(p0, c, azimuth_deg, elevation_deg, domain, time_axis, sensors):
    generated_p0 = generate_p0(p0, azimuth_deg, elevation_deg)
    medium = Medium(domain=domain, sound_speed=c, pml_size=3)
    return simulate_wave_propagation(medium, time_axis, p0=generated_p0, sensors=sensors)

# Define the objective function
def objective_function(p0, c, p_data, azimuth_degs, elevation_degs, domain, time_axis, sensors, lambda1, lambda2):
    num_illuminations = azimuth_degs.shape[0]
    simulated_data = []
    
    for i in range(num_illuminations):
        sim_data = forward_operator(p0, c, azimuth_degs[i], elevation_degs[i], domain, time_axis, sensors)
        simulated_data.append(sim_data)
    
    simulated_data = jnp.stack(simulated_data)
    
    data_fidelity = 0.5 * jnp.sum((simulated_data - p_data)**2)
    regularization = lambda1 * jnp.sum(p0**2) + lambda2 * jnp.sum(jnp.gradient(c)**2)
    
    return data_fidelity + regularization

# Initialize parameters
p0_init = jnp.zeros_like(p_data[0])
c_init = jnp.ones_like(p_data[0])

# Define the gradients
gradients = grad(objective_function, argnums=(0, 1))

# Set up the optimizer
opt_init, opt_update, get_params = adam(step_size=1e-3)
opt_state = opt_init((p0_init, c_init))

@jit
def step(i, opt_state, p_data, azimuth_degs, elevation_degs, domain, time_axis, sensors, lambda1, lambda2):
    p0, c = get_params(opt_state)
    grads = gradients(p0, c, p_data, azimuth_degs, elevation_degs, domain, time_axis, sensors, lambda1, lambda2)
    return opt_update(i, grads, opt_state)

# Perform optimization
num_steps = 1000
for i in range(num_steps):
    opt_state = step(i, opt_state, p_data, azimuth_degs, elevation_degs, domain, time_axis, sensors, lambda1, lambda2)
    if i % 100 == 0:
        p0, c = get_params(opt_state)
        loss = objective_function(p0, c, p_data, azimuth_degs, elevation_degs, domain, time_axis, sensors, lambda1, lambda2)
        print(f'Step {i}, Loss: {loss}')
