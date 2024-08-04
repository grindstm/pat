import os
import signal
import argparse
import time

import jax
import jax.numpy as jnp


import time_reversal
import util as u
import generate_data as gd

jax.clear_caches()


if __name__ == "__main__":
    # Signal handling
    def signal_handler(signum, frame):
        global exit_flag
        exit_flag = True
        print("Exit signal received, finishing current task...")

    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", type=str, default=None, help="data path", nargs="?"
    )
    args = parser.parse_args()
    if args.data_path is not None:
        u.DATA_PATH = args.data_path

    @u.timer
    def reconstruct(file):
        # print current wall time
        # print(time.time())

        file_index = file.split(".")[0]


        # Load the data
        # ----------------------
        sensor_positions = jnp.load(u.file(u.sensors_path, file_index))
        P_data_noisy = jnp.load(u.file(u.P_data_noisy_path, file_index))
        ATT_masks = jnp.load(u.file(u.ATT_masks_path, file_index))
        

        # P_data_file = u.file(u.P_data_path, file_index)
        # P_data = jnp.load(P_data_file)
        # angles_file = u.file(u.angles_path, file_index)
        # angles = jnp.load(angles_file)
        # p0_file = f"{u.DATA_PATH}p0/{file_index}.npy"
        # p0 = jnp.load(p0_file)
        # print(p0_file)
        # c_file = f"{u.DATA_PATH}c/{file_index}.npy"
        # c = jnp.load(c_file)
 
        # ----------------------------------------
        p_rs, c_rs, mses = time_reversal.multi_illumination_parallel_optimized(P_data_noisy, sensor_positions, ATT_masks, num_iterations=u.NUM_ITERATIONS, learning_rate=u.LEARNING_RATE)
        
        # p_rs, c_rs, mses = time_reversal.multi_illumination_parallel(p_data_noisy, sensor_positions, angles, num_iterations=u.NUM_ITERATIONS, learning_rate=u.LEARNING_RATE)
 
        # p_rs, c_rs, mses = time_reversal.multi_illumination(p_data_noisy, sensor_positions, angles, num_iterations=u.NUM_ITERATIONS, learning_rate=u.LEARNING_RATE)
        
        # p_r = time_reversal.lazy_time_reversal(p_data_noisy, sensor_positions)
        # p_rs = [p_r.on_grid]

        # p_rs, mses = time_reversal.iterative_time_reversal(p_data_noisy, sensor_positions, num_iterations=NUM_ITERATIONS, learning_rate=LEARNING_RATE)
        # print(f"Mean squared errors: {mses}")

        # p_r, mses = time_reversal.iterative_time_reversal_optimized(p_data_noisy, sensor_positions, num_iterations=4)
        # ----------------------------------------
        # print(len(p_rs))

        print(f"Mean squared errors: {mses}")
        
        for i, mu_r in enumerate(p_rs):
            if exit_flag:
                break
            mu_r_file = u.file(u.mu_r_path, file_index, i)
            jnp.save(mu_r_file, mu_r.squeeze())
            # print(f"Saved {p_r_file}")

            c_r_file = u.file(u.c_r_path, file_index, i)
            jnp.save(c_r_file, c_rs[i].squeeze())
            # print(f"Saved {c_r_file}")

    for file in os.listdir(u.sensors_path):
        if exit_flag:
            break

        print(f"Processing {file}")
        # p0 files which don't have a corresponding p_r file
        if os.path.exists(u.file(u.mu_r_path, file.split(".")[0])):
            continue

        reconstruct(file)
