import os
import sys
from jwave.geometry import Domain
from jax import numpy as jnp
from jwave.geometry import Medium, Sensors, TimeAxis, BLISensors
from jwave import FourierSeries
from jax import jit
from jwave.acoustics import simulate_wave_propagation

import jax.numpy as jnp
import yaml


# # def main():
params = yaml.safe_load(open("params.yaml"))
N = params["geometry"]["N"]
PML_MARGIN = params["geometry"]["pml_margin"]
DX = params["geometry"]["dx"]
C = params["geometry"]["c"]
CFL = params["geometry"]["cfl"]
NUM_SENSORS = params["simulate"]["num_sensors"]
SENSOR_MARGIN = params["simulate"]["sensor_margin"]



domain = Domain(N, DX)
medium = Medium(domain=domain, sound_speed=C, pml_size=PML_MARGIN)
time_axis = TimeAxis.from_medium(medium, cfl=CFL)
IN_PATH = "data/LNet/"
file = "1_7.npy"
image = jnp.load(IN_PATH + file)
p0_i = jnp.zeros(N)
p0_i = p0_i.at[
    PML_MARGIN : -PML_MARGIN,
    PML_MARGIN : -PML_MARGIN,
    PML_MARGIN : -PML_MARGIN,
].add(image)
p0_i = jnp.zeros(N)
p0 = 1.0 * jnp.expand_dims(p0_i, -1)
p0 = FourierSeries(p0, domain)

sensor_margin = PML_MARGIN + SENSOR_MARGIN
num_sensors_sqrt = jnp.sqrt(NUM_SENSORS).astype(int)
x = jnp.linspace(sensor_margin, N[0] - sensor_margin, num_sensors_sqrt)
y = jnp.linspace(sensor_margin, N[1] - sensor_margin, num_sensors_sqrt)
z = jnp.ones(NUM_SENSORS) * (N[2] - PML_MARGIN)
x, y = jnp.meshgrid(x, y)
sensor_positions = (x.ravel(), y.ravel(), z)
sensors = BLISensors(positions=sensor_positions, n=N)

@jit
def compiled_simulator(p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)
p_data = compiled_simulator(p0)
# # ------------
# # if len(sys.argv) != 3:
# #     print("Usage: python generate_data.py input/path output/path")
# #     sys.exit(1)
# # IN_PATH = sys.argv[1]
# # OUT_PATH = sys.argv[2]

# # os.makedirs(OUT_PATH, exist_ok=True)
# # os.makedirs(f"{OUT_PATH}p0/", exist_ok=True)
# # os.makedirs(f"{OUT_PATH}sensors/", exist_ok=True)


# # for file in os.listdir(IN_PATH):

# # The max file index which doesn't have a corresponding p0 file
# # if os.path.exists(OUT_PATH + f"p0/{file.split('_')[0]}"):
# #     continue
# # file_index = file.split("_")[0]

# # image = jnp.load(IN_PATH + file)

# print(f"Loaded image {file}")

# # Place generated image in the center of the domain

# # Save p0 and sensor positions
# # p0_file = f"{OUT_PATH}p0/{file_index}.npy"
# # sensors_file = f"{OUT_PATH}sensors/{file_index}.npy"
# # print(f"Saving {p0_file} and {sensors_file}")
# # jnp.save(p0_file, p0_i)
# # jnp.save(sensors_file, sensor_positions)

# # print(p0_i.shape)
# p0 = jnp.expand_dims(p0_i, -1)

# print(p_data.shape)

# # p_data_3d = p_data.reshape(
# #     int(time_axis.Nt), int(num_sensors_sqrt), int(num_sensors_sqrt)
# # )
# # p_data_3d = jnp.transpose(p_data_3d, (1, 2, 0))

# # p_data_file = f"{OUT_PATH}p_data/{file_index}.npy"
# # jnp.save(p_data_file, p_data_3d)


# # if __name__ == "__main__":
# #     main()
