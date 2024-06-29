import os
import sys
from jax import numpy as jnp
import jax.numpy as jnp
import yaml
from simulator import Simulator

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    N = params["geometry"]["N"]
    PML_MARGIN = params["geometry"]["pml_margin"]
    DX = params["geometry"]["dx"]
    C = params["geometry"]["c"]
    CFL = params["geometry"]["cfl"]
    NUM_SENSORS = params["simulate"]["num_sensors"]
    SENSOR_MARGIN = params["simulate"]["sensor_margin"]

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
        simulator = Simulator(N, DX, c, CFL)

        simulator.add_pml(image, PML_MARGIN)
        simulator.sensor_plane(NUM_SENSORS, PML_MARGIN, SENSOR_MARGIN)
        simulator.simulate()

        # Save p0, p_data and sensor positions
        p0_file = f"{OUT_PATH}p0/{file_index}.npy"
        jnp.save(p0_file, simulator.p0)
        p_data_file = f"{OUT_PATH}p_data/{file_index}.npy"
        jnp.save(p_data_file, simulator.p_data)
        sensors_file = f"{OUT_PATH}sensors/{file_index}.npy"
        jnp.save(sensors_file, simulator.sensor_positions)
        print(f"Saved {p0_file}, {p_data_file} and {sensors_file}")
