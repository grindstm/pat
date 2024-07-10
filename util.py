import signal
import os
import time
import functools
import sys
import yaml
try:
    import colored_traceback.auto
except ImportError:
    pass

def timer(func):
    """A decorator that prints the execution time of the function it decorates."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()   
        value = func(*args, **kwargs)
        end_time = time.perf_counter()    
        run_time = end_time - start_time 
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def signal_handler(signum, frame):
    global exit_flag
    exit_flag = True
    print("Exit signal received, finishing current task...")

import inspect

def set_globals():
    params = yaml.safe_load(open("params.yaml"))
    # Access the global namespace of the caller
    caller_globals = inspect.stack()[1].frame.f_globals

    # Update the caller's global namespace with parameters
    caller_globals['DATA_PATH'] = params['file']["data_path"]
    caller_globals['BATCH_SIZE'] = params["generate_data"]["batch_size"]
    caller_globals['N'] = tuple(params["geometry"]["N"])
    caller_globals['SHRINK_FACTOR'] = params["geometry"]["shrink_factor"]
    caller_globals['DIMS'] = params["geometry"]["dims"]
    caller_globals['DX'] = tuple(params["geometry"]["dx"])
    caller_globals['C'] = params["geometry"]["c"]
    caller_globals['CFL'] = params["geometry"]["cfl"]
    caller_globals['PML_MARGIN'] = params["geometry"]["pml_margin"]
    caller_globals['TISSUE_MARGIN'] = params["geometry"]["tissue_margin"]
    caller_globals['SENSOR_MARGIN'] = tuple(params["geometry"]["sensor_margin"])
    caller_globals['NUM_SENSORS'] = params["geometry"]["num_sensors"]
    caller_globals['SOUND_SPEED_PERIODICITY'] = params["geometry"]["sound_speed_periodicity"]
    caller_globals['SOUND_SPEED_VARIATION_AMPLITUDE'] = params["geometry"]["sound_speed_variation_amplitude"]
    caller_globals['LIGHTING_ATTENUATION'] = params["lighting"]["lighting_attenuation"]
    caller_globals['NUM_LIGHTING_ANGLES']= params["lighting"]["num_lighting_angles"]
    # caller_globals['AZIMUTH_DEG'] = params["lighting"]["azimuth_deg"]
    # caller_globals['ELEVATION_DEG'] = params["lighting"]["elevation_deg"]
    caller_globals['MU'] = params["lighting"]["mu"]



# # Parse parameters
# def parse_params():
#     params = yaml.safe_load(open("params.yaml"))
#     BATCH_SIZE = params["generate_data"]["batch_size"]
#     N = tuple(params["geometry"]["N"])
#     SHRINK_FACTOR = params["geometry"]["shrink_factor"]
#     DIMS = params["geometry"]["dims"]
#     DX = tuple(params["geometry"]["dx"])
#     C = params["geometry"]["c"]
#     CFL = params["geometry"]["cfl"]
#     PML_MARGIN = params["geometry"]["pml_margin"]
#     TISSUE_MARGIN = params["geometry"]["tissue_margin"]
#     SENSOR_MARGIN = params["geometry"]["sensor_margin"]
#     NUM_SENSORS = params["geometry"]["num_sensors"]
#     SOUND_SPEED_PERIODICITY = params["geometry"]["sound_speed_periodicity"]
#     SOUND_SPEED_VARIATION_AMPLITUDE = params["geometry"][
#         "sound_speed_variation_amplitude"
#     ]
#     LIGHTING_ATTENUATION = params["lighting"]["lighting_attenuation"]
#     AZIMUTH_DEG = params["lighting"]["azimuth_deg"]
#     ELEVATION_DEG = params["lighting"]["elevation_deg"]
#     MU = params["lighting"]["mu"]
#     return BATCH_SIZE, N, SHRINK_FACTOR, DIMS, DX, C, CFL, PML_MARGIN, TISSUE_MARGIN, SENSOR_MARGIN, NUM_SENSORS, SOUND_SPEED_PERIODICITY, SOUND_SPEED_VARIATION_AMPLITUDE, LIGHTING_ATTENUATION, AZIMUTH_DEG, ELEVATION_DEG, MU

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
# os.makedirs(OUT_PATH, exist_ok=True)
# os.makedirs(f"{OUT_PATH}p_r/", exist_ok=True)


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
if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    BATCH_SIZE = params["generate_data"]["batch_size"]
    N = tuple(params["geometry"]["N"])
    DX = tuple(params["geometry"]["dx"])
    C = params["geometry"]["c"]
    CFL = params["geometry"]["cfl"]
    PML_MARGIN = params["geometry"]["pml_margin"]
    TISSUE_MARGIN = params["geometry"]["tissue_margin"]
    SENSOR_MARGIN = params["geometry"]["sensor_margin"]
    NUM_SENSORS = params["geometry"]["num_sensors"]
    SOUND_SPEED_PERIODICITY = params["geometry"]["sound_speed_periodicity"]
    SOUND_SPEED_VARIATION_AMPLITUDE = params["geometry"][
        "sound_speed_variation_amplitude"
    ]






# Reshape p_data to 3D
# p_data_3d = p_data.reshape(
#     int(time_axis.Nt),
#     int(jnp.sqrt(NUM_SENSORS)),
#     int(jnp.sqrt(NUM_SENSORS))
# )
# p_data_3d = jnp.transpose(p_data_3d, (1, 2, 0))

# Save p0, p_data and sensor positions


# def add_gaussian_noise(data, noise_std_dev, key):
#     noise = jax.random.normal(key, shape=data.shape, dtype=data.dtype) * noise_std_dev
#     return data + noise
