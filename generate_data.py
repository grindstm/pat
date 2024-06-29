import os
import sys
import yaml
import numpy as np
from v_system.VSystemGenerator import VSystemGenerator


def main():
    params = yaml.safe_load(open("params.yaml"))
    N = params["geometry"]["N"]
    PML_MARGIN = params["geometry"]["pml_margin"]
    BATCH_SIZE = params["generate_data"]["batch_size"]

    if len(sys.argv) !=2:
        print("Usage: python generate_data.py output/path")
        sys.exit(1)

    # ------------

    sim = VSystemGenerator(
        n=BATCH_SIZE,
        # d0_mean=2.0,
        # d0_std=.50,
        tissue_volume=[n - 2 * PML_MARGIN for n in N],
    )

    images, n_iters = sim.create_networks()

    # ------------

    OUT_PATH = sys.argv[1]
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    print(f"Storing output in {OUT_PATH}")

    folder_index = max([int(filename.split("_")[0]) for filename in os.listdir(OUT_PATH)], default=-1) + 1
 
    # folder_max_n = len(os.listdir(OUT_PATH))
    print(folder_index)

    for i, image in enumerate(images):
        filename = f"{i+folder_index}_{n_iters[i]}"

        np.save(os.path.join(OUT_PATH, filename), image)
        print(f"Created image {filename}")

    # ------------ .


if __name__ == "__main__":
    main()