import os
import sys
from jax import numpy as jnp
import jax.numpy as jnp
import yaml
# from simulator import Simulator

from jax import jit
from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors
from jwave.acoustics import simulate_wave_propagation
params = yaml.safe_load(open("params.yaml"))
N = params["geometry"]["N"]
PML_MARGIN = params["geometry"]["pml_margin"]
DX = params["geometry"]["dx"]
C = params["geometry"]["c"]
CFL = params["geometry"]["cfl"]
NUM_SENSORS = params["simulate"]["num_sensors"]
SENSOR_MARGIN = params["simulate"]["sensor_margin"]


# class Simulator:
# def __init__(N, dx, sound_speed, cfl=0.3):
domain = Domain(N, DX)
sound_speed = 1500. * jnp.ones(N)
medium = Medium(domain=domain, sound_speed=sound_speed)
time_axis = TimeAxis.from_medium(medium, cfl=CFL)
sensor_positions = None
sensors = None
p0 = jnp.empty(N)
p_data = None

def add_pml(image, pml_margin):
    # Place generated image in the center of the domain
    p0_i = jnp.zeros(N)
    p0_i = p0_i.at[
        pml_margin:-pml_margin,
        pml_margin:-pml_margin,
        pml_margin:-pml_margin,
    ].add(image)
    return p0_i

def sensor_plane(num_sensors, pml_margin=10, sensor_margin=10):
    sensor_margin = pml_margin + sensor_margin
    num_sensors_sqrt = jnp.sqrt(num_sensors).astype(int)
    x = jnp.linspace(sensor_margin, N[0] - sensor_margin, num_sensors_sqrt)
    y = jnp.linspace(sensor_margin, N[1] - sensor_margin, num_sensors_sqrt)
    z = jnp.ones(num_sensors) * (N[2] - pml_margin)
    x, y = jnp.meshgrid(x, y)
    sensor_positions = (x.ravel(), y.ravel(), z)
    return BLISensors(positions=sensor_positions, n=N)

@jit
def compiled_simulate(medium, time_axis, p0, sensors):
    return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)


def simulate(p0):
    p0 = jnp.expand_dims(p0, -1)
    p0 = FourierSeries(p0, domain)

    p_data = compiled_simulate(medium, time_axis, p0, sensors)

    p_data_3d = p_data.reshape(
        int(time_axis.Nt),
        int(jnp.sqrt(len(sensors))),
        int(jnp.sqrt(len(sensors))),
    )
    p_data_3d = jnp.transpose(p_data_3d, (1, 2, 0))
    p_data = p_data_3d


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python generate_data.py input/path output/path")
        IN_PATH = "data/LNet/"
        OUT_PATH = "data/"
    else:
        IN_PATH = sys.argv[1]
        OUT_PATH = sys.argv[2]

    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(f"{OUT_PATH}p0/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}sensors/", exist_ok=True)
    os.makedirs(f"{OUT_PATH}p_data/", exist_ok=True)

    for file in os.listdir(IN_PATH):
        # The LNet files which don't have a corresponding p0 file
        if os.path.exists(OUT_PATH + f"p0/{file.split('_')[0]}"):
            continue
        file_index = file.split("_")[0]
        image = jnp.load(IN_PATH + file)
        print(f"Loaded image {file}")

        c = jnp.ones(N) * C
        # simulator = Simulator(N, DX, c, CFL)

        p0_i = add_pml(image, PML_MARGIN)
        sensors = sensor_plane(NUM_SENSORS, PML_MARGIN, SENSOR_MARGIN)
        print(p0.shape)
        p_data = simulate(p0)

        # # Save p0, p_data and sensor positions
        # p0_file = f"{OUT_PATH}p0/{file_index}.npy"
        # jnp.save(p0_file, simulator.p0)
        # p_data_file = f"{OUT_PATH}p_data/{file_index}.npy"
        # jnp.save(p_data_file, simulator.p_data)
        # sensors_file = f"{OUT_PATH}sensors/{file_index}.npy"
        # jnp.save(sensors_file, simulator.sensor_positions)
        # print(f"Saved {p0_file}, {p_data_file} and {sensors_file}")
