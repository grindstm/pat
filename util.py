import signal
import os
import time
import functools
import sys
import yaml

def timer(func):
    """A decorator that prints the execution time of the function it decorates."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1. Start the timer
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2. End the timer
        run_time = end_time - start_time    # 3. Calculate the running time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def signal_handler(signum, frame):
    global exit_flag
    exit_flag = True
    print("Exit signal received, finishing current task...")


# Parse parameters
def parse_params():
    params = yaml.safe_load(open("params.yaml"))
    N = tuple(params["geometry"]["N"])
    PML_MARGIN = params["geometry"]["pml_margin"]
    DX = tuple(params["geometry"]["dx"])
    C = params["geometry"]["c"]
    CFL = params["geometry"]["cfl"]
    NUM_SENSORS = params["simulate"]["num_sensors"]
    SENSOR_MARGIN = params["simulate"]["sensor_margin"]
    return N, PML_MARGIN, DX, C, CFL, NUM_SENSORS, SENSOR_MARGIN

# Parse arguments
def parse_args():
    if len(sys.argv) == 2:
        IN_PATH = sys.argv[1]
        OUT_PATH = sys.argv[1]
    elif len(sys.argv) == 3:
        IN_PATH = sys.argv[1]
        OUT_PATH = sys.argv[2]
    else:
        IN_PATH = "data/"
        OUT_PATH = "data/"
    return IN_PATH, OUT_PATH
    # Output directories
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(f"{OUT_PATH}p_r/", exist_ok=True)


def loop(func):
    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)
    for file in os.listdir(f"{IN_PATH}p0/"):
        if exit_flag:
            break
        
        print(f"Processing {file}")
        # p0 files which don't have a corresponding p_r file
        if os.path.exists(OUT_PATH + f"p_r/{file.split('.')[0]}"):
            continue

        func(file)

# Reshape p_data to 3D
# p_data_3d = p_data.reshape(
#     int(time_axis.Nt),
#     int(jnp.sqrt(NUM_SENSORS)),
#     int(jnp.sqrt(NUM_SENSORS))
# )
# p_data_3d = jnp.transpose(p_data_3d, (1, 2, 0))

# Save p0, p_data and sensor positions