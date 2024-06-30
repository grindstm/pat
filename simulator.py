from jax import jit
from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation
from functools import partial

@jit
def compiled_simulate(medium, time_axis, p0, sensors):
    return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)

class Simulator:
    def __init__(self, N, dx, sound_speed, cfl=0.3, pml_margin=10):
        self.N = N
        self.dx = dx
        self.domain = Domain(N, dx)
        self.sound_speed = sound_speed
        self.pml_margin = pml_margin
        self.medium = Medium(domain=self.domain, sound_speed=self.sound_speed, pml_size=self.pml_margin)
        self.time_axis = TimeAxis.from_medium(self.medium, cfl=cfl)
        self.sensor_positions = None
        self.sensors = None
        self.p0 = jnp.empty(self.N)
        self.p_data = None

    def add_pml(self, image, pml_margin):
        # Place generated image in the center of the domain
        p0_i = jnp.zeros(self.N)
        p0_i = p0_i.at[
            pml_margin:-pml_margin,
            pml_margin:-pml_margin,
            pml_margin:-pml_margin,
        ].add(image)
        self.p0 = p0_i

    def sensor_plane(self, num_sensors, pml_margin=10, sensor_margin=10):
        sensor_margin = pml_margin + sensor_margin
        num_sensors_sqrt = jnp.sqrt(num_sensors).astype(int)
        x = jnp.linspace(sensor_margin, self.N[0] - sensor_margin, num_sensors_sqrt)
        y = jnp.linspace(sensor_margin, self.N[1] - sensor_margin, num_sensors_sqrt)
        z = jnp.ones(num_sensors) * (self.N[2] - pml_margin)
        x, y = jnp.meshgrid(x, y)
        self.sensor_positions = (x.ravel(), y.ravel(), z)
        self.sensors = BLISensors(positions=self.sensor_positions, n=self.N)

    def simulate(self):
        p0 = 1.0 * jnp.expand_dims(self.p0, -1)
        p0 = FourierSeries(p0, self.domain)
        print(p0.on_grid.shape)

        p_data = compiled_simulate(self.medium, self.time_axis, p0, self.sensors)

        p_data_3d = p_data.reshape(
            int(self.time_axis.Nt),
            int(jnp.sqrt(len(self.sensors))),
            int(jnp.sqrt(len(self.sensors))),
        )
        p_data_3d = jnp.transpose(p_data_3d, (1, 2, 0))
        self.p_data = p_data_3d

