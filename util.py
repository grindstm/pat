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

# environment variable to prevent jax preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"


def timer(func):
    """A decorator that prints the execution time of the function it decorates."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value, run_time

    return wrapper_timer


params = yaml.safe_load(open("params.yaml"))
DATA_PATH = params["file"]["data_path"]
BATCH_SIZE = params["generate_data"]["batch_size"]
N = tuple(params["geometry"]["N"])
SHRINK_FACTOR = params["geometry"]["shrink_factor"]
DIMS = params["geometry"]["dims"]
DX = tuple(params["geometry"]["dx"])
C = params["geometry"]["c"]
C_PERIODICITY = params["geometry"]["c_periodicity"]
C_VARIATION_AMPLITUDE = params["geometry"]["c_variation_amplitude"]
C_BLOOD = params["geometry"]["c_blood"]
CFL = params["geometry"]["cfl"]
PML_MARGIN = params["geometry"]["pml_margin"]
TISSUE_MARGIN = params["geometry"]["tissue_margin"]
SENSOR_MARGIN = tuple(params["geometry"]["sensor_margin"])
NUM_SENSORS = params["geometry"]["num_sensors"]
LIGHTING_ATTENUATION = params["lighting"]["lighting_attenuation"]
NUM_LIGHTING_ANGLES = params["lighting"]["num_lighting_angles"]
ATTENUATION = params["lighting"]["attenuation"]

NOISE_AMPLITUDE = params["reconstruct"]["noise_amplitude"]
RECON_ITERATIONS = params["reconstruct"]["recon_iterations"]
LR_MU_R = params["reconstruct"]["lr_mu_r"]
LR_C_R = params["reconstruct"]["lr_c_r"]
RECON_FILE_START = params["reconstruct"]["recon_file_start"]
RECON_FILE_END = params["reconstruct"]["recon_file_end"]

CHECKPOINT_FILE_INDEX = params["train"]["checkpoint_index"]
LR_R_MU = params["train"]["lr_R_mu"]
LR_R_C = params["train"]["lr_R_c"]
DROPOUT = params["train"]["dropout"]
TRAIN_FILE_START = params["train"]["train_file_start"]
TRAIN_FILE_END = params["train"]["train_file_end"]

# metrics_path = os.path.join(DATA_PATH, "metrics")
mu_path = os.path.join(DATA_PATH, "mu")
angles_path = os.path.join(DATA_PATH, "angles")
ATT_masks_path = os.path.join(DATA_PATH, "ATT_masks")
p0_path = os.path.join(DATA_PATH, "P_0")
c_path = os.path.join(DATA_PATH, "c")
sensors_path = os.path.join(DATA_PATH, "sensors")
P_0_path = os.path.join(DATA_PATH, "P_0")
P_data_path = os.path.join(DATA_PATH, "P_data")
P_data_noisy_path = os.path.join(DATA_PATH, "P_data_noisy")
mu_r_path = os.path.join(DATA_PATH, "mu_r")
c_r_path = os.path.join(DATA_PATH, "c_r")
params_R_mu_path = os.path.join(DATA_PATH, "checkpoints", "params_R_mu")
params_R_c_path = os.path.join(DATA_PATH, "checkpoints", "params_R_c")
checkpoints_path = os.path.join(DATA_PATH, "checkpoints", str(CHECKPOINT_FILE_INDEX))
state_path = os.path.join(DATA_PATH, "state")

os.makedirs(DATA_PATH, exist_ok=True)
# os.makedirs(metrics_path, exist_ok=True)

os.makedirs(mu_path, exist_ok=True)
os.makedirs(angles_path, exist_ok=True)
os.makedirs(c_path, exist_ok=True)
os.makedirs(P_data_path, exist_ok=True)
os.makedirs(P_data_noisy_path, exist_ok=True)
os.makedirs(sensors_path, exist_ok=True)
os.makedirs(P_0_path, exist_ok=True)
os.makedirs(mu_r_path, exist_ok=True)
os.makedirs(c_r_path, exist_ok=True)
os.makedirs(ATT_masks_path, exist_ok=True)


def file(path, index, iteration=None):
    if iteration is not None:
        return os.path.join(path, f"{index}_{iteration}.npy")
    else:
        return os.path.join(path, f"{index}.npy")


def max_file_index(path):
    return (
        max(
            [
                int(filename.split("_")[0])
                for filename in os.listdir(path)
                if filename.split("_")[0].isdigit()
            ],
            default=-1,
        )
        + 1
    )


def max_iteration():
    """
    Returns the maximum iteration number in the mu_r_path directory.
    """
    if not os.path.exists(mu_r_path):
        return 0
    else:
        return max(
            [
                int(filename.split("_")[1].split(".")[0])
                for filename in os.listdir(mu_r_path)
                if filename.split("_")[1].split(".")[0].isdigit()
            ],
            default=0,
        )


# Reshape p_data to 3D
# p_data_3d = p_data.reshape(
#     int(time_axis.Nt),
#     int(jnp.sqrt(NUM_SENSORS)),
#     int(jnp.sqrt(NUM_SENSORS))
# )
# p_data_3d = jnp.transpose(p_data_3d, (1, 2, 0))


# def add_gaussian_noise(data, noise_std_dev, key):
#     noise = jax.random.normal(key, shape=data.shape, dtype=data.dtype) * noise_std_dev
#     return data + noise
