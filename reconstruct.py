import os
import pickle
import shutil
import signal
from collections import defaultdict
import argparse
from typing import Any
from functools import partial
from typing import TypeVar, Mapping
import numpy as np

import jax
from jax import random
from jax import vmap, jit, value_and_grad
import jax.profiler
import jax.numpy as jnp
from jax import tree_util

from flax import linen as nn
from flax import struct 
from flax.training import train_state
from flax.training import orbax_utils
import flax.serialization
import msgpack

import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_utils

from jwave.geometry import Domain, Medium, BLISensors, TimeAxis
from jwave import simulate_wave_propagation

from jaxdf import FourierSeries
from jaxdf.operators import compose
from jaxdf.operators.differential import laplacian, gradient, diag_jacobian
from jaxdf.operators.functions import compose, sum_over_dims

import util as u
from PADataset import PADataset

jax.clear_caches()

# --------------------------------------------
# Utilities and constants
# --------------------------------------------
def signal_handler(signum, frame):
    global exit_flag
    exit_flag = True
    print("Exit signal received, finishing current task...")

def save_recon(j, recon):
    """
    Save the reconstruction results to disk.
    
    Args:
        j (int): File index.
        recon (dict): Dictionary containing the recon
    """
    iteration_count = len(recon["mu_rs"])
    for i in range(iteration_count):
        mu_r_file = u.file(u.mu_r_path, j, i)
        jnp.save(mu_r_file, recon["mu_rs"][i].squeeze())
        c_r_file = u.file(u.c_r_path, j, i)
        jnp.save(c_r_file, recon["c_rs"][i].squeeze())

def save_state(checkpoint_path, step, state, keep=4):
    """
    Save the training state to disk.
    
    Args:
        checkpoint_path (str): Path to the checkpoint folder.
        step (int): Current step.
        state (dict): Dictionary containing the training state.
        keep (int): Number of checkpoints to keep.
    """
    all_steps = sorted(int(f.split('.')[0]) for f in os.listdir(checkpoint_path) if f.endswith('.msgpack'))
    if len(all_steps) > keep:
        oldest_step = all_steps[0]
        os.remove(os.path.join(checkpoint_path, f"{oldest_step}.msgpack"))

    state_to_save = {
        "state_r": [flax.serialization.to_bytes(train_state) for train_state in state["state_r"]],
        "losses_batch": state["losses_batch"]
    }
    with open(os.path.join(checkpoint_path, f"{step}.msgpack"), "wb") as f:
        f.write(msgpack.packb(state_to_save, use_bin_type=True))

def restore_state(checkpoint_path, latest_step):
    """
    Restore the training state from disk.

    Args:
        checkpoint_path (str): Path to the checkpoint folder.
        latest_step (int): Latest step.

    Returns:
        dict: Dictionary containing the training state.
    """
    checkpoint_file = os.path.join(checkpoint_path, f"{latest_step}.msgpack")
    with open(checkpoint_file, "rb") as f:
        state_packed = msgpack.unpackb(f.read(), strict_map_key=False)
    
    state = {
        "state_r": [flax.serialization.from_bytes(TrainState, train_state_bytes) for train_state_bytes in state_packed["state_r"]],
        "losses_batch": state_packed["losses_batch"]
    }
    return state

def get_latest_checkpoint_step(checkpoint_path):
    """
    Get the latest checkpoint step from disk.

    Args:
        checkpoint_path (str): Path to the checkpoint folder.

    Returns:
        int: Latest step.
    """
    checkpoint_files = []
    if os.path.exists(checkpoint_path):
        checkpoint_files = [f for f in os.listdir(checkpoint_path)]
    if len(checkpoint_files) == 0:
        return None
    steps = [int(f.split(".")[0]) for f in checkpoint_files]
    return max(steps)

def print_losses(j, state):
    print(f"File {j}\nIter\tLoss_data\tLoss_mu\tLoss_c\tLoss_sum_mu_c")
    for i in range(u.RECON_ITERATIONS):
        print(
            f"{i}\t{state.losses[j]["data"][i]:.6f}\t{state.losses[j]["mu"][i]:.6f}\t{state.losses[j]["c"][i]:.6f}\t{state.losses[j]["sum_mu_c"][i]:.6f}"
    )

def print_recon_losses(j, losses, num_to_print=5):
    print(f"File {j} \nIter\tLoss_data\tLoss_mu\t\tLoss_c\t\tLoss_sum_mu_c")
    iteration_count = len(losses["data"])
    indices = np.linspace(0, iteration_count - 1, num_to_print, dtype=int)
    for i in indices:
        print(
            f"{i}\t{losses['data'][i]:.6f}\t{losses['mu'][i]:.6f}\t{losses['c'][i]:.6f}\t{losses['sum_mu_c'][i]:.6f}"
        )

def print_net(net, shapes, extra_args={}):
    inputs = (jnp.ones(shapes[i]) for i in range(len(shapes)))
    print(
        net.tabulate(
            jax.random.key(0),
            *inputs,
            **extra_args,
            compute_flops=True,
            compute_vjp_flops=True)
        )

if u.DIMS == 2:
    N = u.N[:2]
    DX = u.DX[:2]

P0_shape = (u.NUM_LIGHTING_ANGLES, *N, 1)
im_shape = (1, *N, 1)
domain = Domain(N, DX)
medium = Medium(domain=domain, sound_speed=jnp.ones(N) * u.C, pml_size=u.PML_MARGIN[0])
time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)

sensors_obj = BLISensors(positions=np.load(u.file(u.sensors_path, 0)), n=domain.N)

def simulate(medium, time_axis, p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)

batch_compiled_simulate = vmap(simulate, in_axes=(None, None, 0))

@jit
def mse(x, x_true):
    return jnp.mean(jnp.square(x - x_true)) / 2.0

def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()

def get_sound_speed(params):
    """
    Get the sound speed from the parameterized field.
    """
    return 1420.0 + 150.0*compose(params)(nn.sigmoid)

def get_mu(params):
    """
    Get the attenuation coefficient from the parameterized field.
    """
    return compose(params)(nn.softplus)

# Anisotropic Diffusion Functions
# Divergence Operator
def divergence(u, stagger):
    return sum_over_dims(diag_jacobian(u, stagger=stagger))

# Diffusion Conductivity Kernel
def conductivity_kernel(u):
    kernel = lambda x: 1 / (1 + (x / 0.03) ** 2)
    return compose(u)(kernel)

# Gradient Magnitude
def norm(u):
    z = sum_over_dims(u ** 2)
    return compose(z)(jnp.sqrt)

# Anisotropic Diffusion Function
@jit
def anisotropic_diffusion(u, stagger=[0.5]):
    grad_u = gradient(u, stagger=stagger)
    mod_gradient = norm(grad_u)
    c = conductivity_kernel(mod_gradient)
    return divergence(c * grad_u, stagger=[-0.5])


# --------------------------------------------
# Network definitions
# --------------------------------------------
class ConvBlock(nn.Module):
    dropout: float
    features: int = None
    activation: str = "relu"
    train: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME", kernel_init=nn.initializers.he_normal())(x)
        x = nn.BatchNorm(use_running_average=not self.train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not self.train)

        return x

class EncoderBlock(nn.Module):
    features: int
    dropout: float
    activation: str = "relu"
    train: bool = True

    @nn.compact
    def __call__(self, x):
        Conver = partial(
            ConvBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )

        x = Conver(features=self.features)(x)
        x = Conver(features=self.features)(x)

        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        return x

class DecoderBlock(nn.Module):
    features: int
    dropout: float
    activation: str = "relu"
    train: bool = True

    @nn.compact
    def __call__(self, x):
        Conver = partial(
            ConvBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )

        x = jax.image.resize(
            x,
            (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]),
            method="bilinear",
        )

        x = Conver(features=self.features)(x)
        x = Conver(features=self.features)(x)

        return x

class TreeNet(nn.Module):
    """
    This network combines 4 fields into a single field output using skip connections. All fields must have the same batch size. As a convention, the expected fields are:
    x0: mu_r
    x1: d_mu
    x2: c_r
    x3: d_c  
    
    output: the modified gradient d_c_r
    """
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x0, x1, x2, x3, train: bool = True):
        f = self.features
        Encoder = partial(
            EncoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Conver = partial(
            ConvBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Decoder = partial(
            DecoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        # 
        x0 = jnp.broadcast_to(x2, (x0.shape))
        # 
        e0_1 = Encoder(features=f * 2)(x0)
        e0_2 = Encoder(features=f * 4)(e0_1)    
        e0_3 = Encoder(features=f * 8)(e0_2)

        e1_1 = Encoder(features=f * 2)(x1)
        e1_2 = Encoder(features=f * 4)(e1_1)
        e1_3 = Encoder(features=f * 8)(e1_2)

        e2_1 = Encoder(features=f * 2)(x2)
        e2_2 = Encoder(features=f * 4)(e2_1)
        e2_3 = Encoder(features=f * 8)(e2_2)

        e3_1 = Encoder(features=f * 2)(x3)
        e3_2 = Encoder(features=f * 4)(e3_1)
        e3_3 = Encoder(features=f * 8)(e3_2)
               
        c = jnp.concatenate([e0_3, e1_3, e2_3, e3_3], axis=-1)

        c = Conver(features=f * 8)(c)
        c = Conver(features=f * 8)(c)

        d2 = Decoder(features=f * 4)(c)
        d2 = jnp.concatenate([d2, e0_2, e1_2, e2_2, e3_2], axis=-1)
        d1 = Decoder(features=f * 2)(d2)
        d1 = jnp.concatenate([d1, e0_1, e1_1, e2_1, e3_1], axis=-1)
        d0 = Decoder(features=f)(d1)

        o = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(d0)

        return o
class TreeNet_P0(nn.Module):
    """
    This network combines 4 fields into a single field output using skip connections. The first field accommodates a batch (of illuminations). As a convention, the expected fields are:
    x0: P0_r
    x1: d_mu
    x2: c_r
    x3: d_c 

    output: the modified gradient d_c_r
    """
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x0, x1, x2, x3, train: bool = True):
        f = self.features
        Encoder = partial(
            EncoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Conver = partial(
            ConvBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Decoder = partial(
            DecoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        # 
        x0 = jnp.broadcast_to(x2, (x0.shape))
        # 
        e0_1 = Encoder(features=f * 2)(x0)
        e0_2 = Encoder(features=f * 4)(e0_1)    
        e0_3 = Encoder(features=f * 8)(e0_2)

        e1_1 = Encoder(features=f * 2)(x1)
        e1_2 = Encoder(features=f * 4)(e1_1)
        e1_3 = Encoder(features=f * 8)(e1_2)

        e2_1 = Encoder(features=f * 2)(x2)
        e2_2 = Encoder(features=f * 4)(e2_1)
        e2_3 = Encoder(features=f * 8)(e2_2)

        e3_1 = Encoder(features=f * 2)(x3)
        e3_2 = Encoder(features=f * 4)(e3_1)
        e3_3 = Encoder(features=f * 8)(e3_2)
        
        e1_1 = jnp.broadcast_to(e1_1, e0_1.shape)
        e2_1 = jnp.broadcast_to(e2_1, e0_1.shape)
        e3_1 = jnp.broadcast_to(e3_1, e0_1.shape)
        
        e1_2 = jnp.broadcast_to(e1_2, e0_2.shape)
        e2_2 = jnp.broadcast_to(e2_2, e0_2.shape)
        e3_2 = jnp.broadcast_to(e3_2, e0_2.shape)
        
        e1_3 = jnp.broadcast_to(e1_3, e0_3.shape)
        e2_3 = jnp.broadcast_to(e2_3, e0_3.shape)
        e3_3 = jnp.broadcast_to(e3_3, e0_3.shape)
       
        c = jnp.concatenate([e0_3, e1_3, e2_3, e3_3], axis=-1)

        c = Conver(features=f * 8)(c)
        c = Conver(features=f * 8)(c)

        d2 = Decoder(features=f * 4)(c)
        d2 = jnp.concatenate([d2, e0_2, e1_2, e2_2, e3_2], axis=-1)
        d1 = Decoder(features=f * 2)(d2)
        d1 = jnp.concatenate([d1, e0_1, e1_1, e2_1, e3_1], axis=-1)
        d0 = Decoder(features=f)(d1)

        o = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(d0)
        o = jnp.permute_dims(o, (3, 1, 2, 0)) 
        o = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o)
    
        return o
   
class YNet(nn.Module):
    """
    This network combines 2 fields into a single field output using skip connections. All fields must have the same batch size. As a convention, the expected fields are:
    x0: d_mu
    x1: d_c

    output: the modified gradient d_c_r
    """
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x0, x1, train: bool = True):
        f = self.features
        Encoder = partial(
            EncoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Conver = partial(
            ConvBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Decoder = partial(
            DecoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )

        e0_1 = Encoder(features=f * 2)(x0)
        e0_2 = Encoder(features=f * 4)(e0_1)    
        e0_3 = Encoder(features=f * 8)(e0_2)

        e1_1 = Encoder(features=f * 2)(x1)
        e1_2 = Encoder(features=f * 4)(e1_1)
        e1_3 = Encoder(features=f * 8)(e1_2)
        
        c = jnp.concatenate([e0_3, e1_3], axis=-1)

        c = Conver(features=f * 8)(c)
        c = Conver(features=f * 8)(c)

        d2 = Decoder(features=f * 4)(c)
        d2 = jnp.concatenate([d2, e0_2, e1_2], axis=-1)
        d1 = Decoder(features=f * 2)(d2)
        d1 = jnp.concatenate([d1, e0_1, e1_1], axis=-1)
        d0 = Decoder(features=f)(d1)

        o0 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(d0)

        return o0


class ConcatNet(nn.Module):
    """
    This network combines 4 fields into 2 outputs without skip connections. All fields must have the same batch size. As a convention, the expected fields are:
    x0: P0_r
    x1: d_P0
    x2: c_r
    x3: d_c

    output: the modified gradients d_c_r, d_P0_r
    """

    features: int
    dropout: float
    activation: str = "relu"
    train: bool = True

    @nn.compact
    def __call__(self, x0, x1, x2, x3):
        f = self.features
        Encoder = partial(
            EncoderBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )
        Conver = partial(
            ConvBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )
        Decoder = partial(
            DecoderBlock,
            dropout=self.dropout,
            activation=self.activation,
            train=self.train,
        )

        x2_b = jnp.broadcast_to(x2, (x0.shape))
        x3_b = jnp.broadcast_to(x3, (x0.shape))

        x0 = Conver(features=f // 2)(x0)
        x1 = Conver(features=f // 2)(x1)
        x2_b = Conver(features=f // 2)(x2_b)
        x3_b = Conver(features=f // 2)(x3_b)

        c = jnp.concatenate([x0, x1, x2_b, x3_b], axis=-1)

        x = Encoder(features=f * 2)(c)
        x = Encoder(features=f * 4)(x)
        x = Encoder(features=f * 8)(x)

        x = Conver(features=f * 8)(x)

        x = jnp.split(x, 2, axis=-1)

        o0 = Conver(features=f * 8)(x[0])
        o1 = Conver(features=f * 8)(x[1])

        o0 = Decoder(features=f * 8)(o0)
        o0 = Decoder(features=f * 4)(o0)
        o0 = Decoder(features=f * 2)(o0)

        o1 = Decoder(features=f * 8)(o1)
        o1 = Decoder(features=f * 4)(o1)
        o1 = Decoder(features=f * 2)(o1)

        return o0, o1

class StepNet(nn.Module):
    """
    This network combines the 4 fields into 2 outputs.
    x_next = x_previous - alpha * dx - R(x)
    where R is a network defined above.
    """
    features: int
    dropout: float
    network: nn.Module
    activation: str = "relu"

    @nn.compact
    def __call__(self, x0, x1, x2, x3, train: bool = True):
        """
        from iteration k:
        x0: mu_r
        x1: d_mu
        x2: c_r
        x3: d_c

        outputs iteration k+1:
        o0: mu_r
        o1: d_mu
        """
        R = self.network(
            features=self.features,
            dropout=self.dropout,
            activation=self.activation,
            train=train,
        )
        r = R(x0, x1, x2, x3)
   
        alpha_0 = self.param("alpha_0", nn.initializers.ones, ())
        o0 = x0 - alpha_0 * x1 - r[0]
        o0 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o0)
        o0 = jnp.permute_dims(o0, (3, 1, 2, 0))
        o0 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o0)

        alpha_1 = self.param("alpha_1", nn.initializers.ones, ())
        o1 = x2 - alpha_1 * x3 - r[1]
        o1 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o1)
        o1 = jnp.permute_dims(o1, (3, 1, 2, 0))
        o1 = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(o1)

        return o0, o1
    
# --------------------------------------------
# Initialize Network and train state
# --------------------------------------------
class TrainState(train_state.TrainState):
    key: jax.Array
    batch_stats: Any
    losses = defaultdict(dict)

    
def create_train_state(
    key,
    model,
    learning_rate,
    shapes,
    num_steps,
):
    """
    Create a list of training states for the given model. Useful when using different parameters for each iteration.

    Args:
        key (jax.Array): Random key.
        model (nn.Module): Model to train.
        learning_rate (float): Learning rate.
        shapes (list): List of shapes for the model inputs.
        num_steps (int): Number of training steps.

    Returns:
        list: List of training states
    """
    states = []
    key, key_init, *keys = random.split(key, 2 + num_steps)

    for step in range(num_steps):
        inputs = (random.normal(keys[i], shapes[i]) for i in range(len(shapes)))

        variables = model.init(
            key_init,
            *inputs,
        )

        batch_stats = variables.get("batch_stats", None)
        tx = optax.adam(learning_rate)
        
        train_state = TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=tx,
            batch_stats=batch_stats,
            key=keys[step] 
        )

        states.append(train_state)

    return states
# --------------------------------------------
features_0 = 32
# R = StepNet(features=features_0, dropout=0.1, activation="elu", network=ConcatNet)
# R_mu = YNet(features=features_0, dropout=u.DROPOUT, activation="relu")
# R = YNet(features=features_0, dropout=u.DROPOUT, activation="relu")
R = TreeNet(features=features_0, dropout=u.DROPOUT, activation="relu")
# --------------------------------------------


def train_r_c(num_illum=4, lr=[u.LR_MU_R, u.LR_C_R], num_it=u.RECON_ITERATIONS, cont=False):
    """
    Train the c regularizer. Writes the reconstruction results and last 4 checkpoints to disk.

    Args:
        num_illum (int): Number of illuminations to use. Default is 4.
        lr (list): Learning rates for mu and c. Default is [u.LR_MU_R, u.LR_C_R].
        num_it (int): Number of iterations. Default is u.RECON_ITERATIONS.
        cont (bool): Continue training from the latest checkpoint. Default is False.

    Returns:
        None
    """
    global exit_flag
    jax.clear_caches()

    key = random.PRNGKey(59)
    key, key_train_state = random.split(key)

    dataset = PADataset()

    state_r = create_train_state(
        key_train_state,
        R,
        learning_rate=u.LR_R_C,
        shapes=[im_shape, im_shape, im_shape, im_shape],
        num_steps=num_it
    )

    # Checkpoint restore
    # ------------------
    checkpoint_path = u.checkpoints_path

    latest_step = get_latest_checkpoint_step(checkpoint_path)
    if cont and latest_step is not None:
        try:
            restored = restore_state(checkpoint_path, latest_step)
            # state_r_mu = restored["r_mu"]
            state_r = restored["r"]
            # losses_batch = restored["losses"]
            
            print(f"Restored checkpoint {checkpoint_path}/{latest_step}")
        except Exception as e:
            print(f"Couldn't load checkpoint {checkpoint_path}/{latest_step}\n Run again without -c flag to start from scratch")
            print(f"Error: {e}")
            exit_flag = True
    else:
        shutil.rmtree(checkpoint_path, ignore_errors=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        print("Created empty checkpoint folder")
        losses_batch= defaultdict(dict)

    # ------------------

    start = latest_step if cont else u.TRAIN_FILE_START
    for file_index in range(start, u.TRAIN_FILE_END):
        if exit_flag:
            break

        # Illumination angles
        # -------------------
        num_angles = dataset.num_angles
        illum_indices = np.linspace(0, num_angles, num_illum, endpoint=False).astype(int)  if num_illum < np.inf else np.array(range(num_angles))
        data = dataset[(file_index, illum_indices)]
        print(f"illumination angles: {data["angles"]}")
        # -------------------

        j = data["file_idx"]
        mu = data["mu"]
        ATT_masks = FourierSeries(data["ATT_masks"], domain)
        c = data["c"]
        P_data = data["P_data"]

        losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
        recon = defaultdict(dict)

        c_p = FourierSeries(jnp.zeros(im_shape)-5, domain) 
        mu_p = FourierSeries(jnp.zeros(im_shape)+1, domain) 

        opt_mu = optax.adam(learning_rate=lr[0])
        opt_c = optax.adam(learning_rate=lr[1])
        opt_mu_state = opt_mu.init(mu_p)
        opt_c_state = opt_c.init(c_p)

        mu_rs = []
        c_rs = []

        # @jit
        def loss_rec(mu_p, c_p):
            mu_r = get_mu(mu_p)
            c_r = get_sound_speed(c_p)
            P0 = mu_r * ATT_masks
            medium = Medium(domain=domain, sound_speed=c_r[0], pml_size=u.PML_MARGIN[0])
            P_pred = batch_compiled_simulate(medium, time_axis, P0)
            return mse(P_pred.squeeze(), P_data)
        
        for i in range(num_it):
            
            loss_data, (d_mu_p, d_c_p) = value_and_grad(loss_rec, (0, 1))(mu_p, c_p)

            updates_mu, opt_mu_state = opt_mu.update(d_mu_p, opt_mu_state)
            mu_p = optax.apply_updates(mu_p, updates_mu)
            mu_r = get_mu(mu_p)
            
            if i % 2 == 0: # Share gradients every other iteration
                updates_c, opt_c_state = opt_c.update(d_mu_p, opt_c_state)
                c_p = optax.apply_updates(c_p, updates_c)
                c_r = get_sound_speed(c_p)

            else:
                def loss_fn(r_p, mu_p, d_mu_p, c_p, d_c_p, opt_c_state, batch_stats):
                    d_c_p, updates = state_r[i].apply_fn(
                        {'params': r_p, 'batch_stats': batch_stats},
                        mu_p.on_grid,
                        d_mu_p.on_grid,
                        c_p.on_grid,
                        d_c_p.on_grid,
                        train=True,
                        rngs={"dropout": key},
                        mutable=["batch_stats"]
                    )

                    updates_c, opt_c_state = opt_c.update(d_c_p, opt_c_state)
                    c_p = optax.apply_updates(c_p, updates_c)

                    c_r = get_sound_speed(c_p)

                    loss_r = mse(c_r.on_grid, c)

                    return loss_r, (c_r, c_p, opt_c_state, updates)

                (loss_r, (c_r, c_p, opt_c_state, updates)), d_r_p = value_and_grad(loss_fn, (0), has_aux=True)(state_r[i].params, mu_p, d_mu_p, c_p, d_c_p, opt_c_state, state_r[i].batch_stats)

                state_r[i] = state_r[i].apply_gradients(grads=d_r_p, batch_stats=updates)    
            
                print(f"loss_r: {loss_r:.4f}")

            losses["data"].append(float(loss_data))
            mu_rs.append(mu_r.on_grid.squeeze())
            c_rs.append(c_r.on_grid.squeeze())
            losses["c"].append(float(mse(c_rs[-1], c)/jnp.mean(c)))
            losses["mu"].append(float(mse(mu_rs[-1], mu)/jnp.mean(mu)))
            losses["sum_mu_c"].append(float(losses["c"][-1] + losses["mu"][-1]))
        
        recon["mu_rs"] = mu_rs
        recon["c_rs"] = c_rs
        losses[j] = losses
        save_recon(j, recon)
        print_recon_losses(j, losses)

        state={"state_r": state_r, "losses_batch": losses_batch}

        save_state(checkpoint_path, file_index, state, keep=4)


@jit
def recon_step(mu_p, c_p, ATT_masks, P_data):
    """
    Data fidelity gradient calculation for the reconstruction.

    Args:
        mu_p (FourierSeries): Attenuation coefficient.
        c_p (FourierSeries): Sound speed.
        ATT_masks (FourierSeries): Attenuation masks.
        P_data (jnp.array): Data to reconstruct.

    Returns:
        tuple: Loss and gradients.
    """
    def loss_rec(mu_p, c_p):
        mu_r = get_mu(mu_p)
        c_r = get_sound_speed(c_p)
        P0 = mu_r * ATT_masks
        medium = Medium(domain=domain, sound_speed=c_r[0], pml_size=u.PML_MARGIN[0])
        P_pred = batch_compiled_simulate(medium, time_axis, P0)
        return mse(P_pred.squeeze(), P_data)
    loss_data, (d_mu_p, d_c_p) = value_and_grad(loss_rec, (0, 1))(mu_p, c_p)
    
    return loss_data, d_mu_p, d_c_p


def recon_2opt_r(dataset, file_index, func_step, lr=[1.,1.], num_illum=np.inf, num_it=u.RECON_ITERATIONS, func_step_kwargs={}):
    """
    Calls a reconstruction function using 2 optimizers and gradient sharing, with learned regularization on the c gradients. Optionally, it can be limited to a number of illuminations.

    If performing multiple reconstructions, it may be preferrable to call recon_batch, feeding this as the func_recon parameter instead.

    Args:
        dataset (PADataset): Dataset to use.
        file_index (int): File index to reconstruct.
        func_step (function): Reconstruction function.
        lr (list): Learning rates for mu and c. Default is [1., 1.].
        num_illum (int): Number of illuminations to use. Default is np.inf.
        num_it (int): Number of iterations. Default is u.RECON_ITERATIONS.
        func_step_kwargs (dict): Additional arguments for the reconstruction function.

    Returns:
        tuple: Reconstruction losses and results.
    """
    jax.clear_caches()
    num_angles = dataset.num_angles
    illum_indices = np.linspace(0, num_angles, num_illum, endpoint=False).astype(int)  if num_illum < np.inf else np.array(range(num_angles))
    data = dataset[(file_index, illum_indices)]

    print(f"illumination angles: {data["angles"]}")

    j = data["file_idx"]
    mu = data["mu"]
    ATT_masks = FourierSeries(data["ATT_masks"], domain)
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    c_p = FourierSeries(jnp.zeros(im_shape)-5, domain) 
    mu_p = FourierSeries(jnp.zeros(im_shape)+1, domain) 

    opt_mu = optax.adam(learning_rate=lr[0])
    opt_c = optax.adam(learning_rate=lr[1])
    opt_mu_state = opt_mu.init(mu_p)
    opt_c_state = opt_c.init(c_p)

    try:
        latest_step = get_latest_checkpoint_step(u.checkpoints_path)
        restored = restore_state(u.checkpoints_path, latest_step)
        state_r = restored["state_r"]
        print(f"Restored checkpoint {u.checkpoints_path}/{latest_step}")
    except Exception as e:
        print(f"Couldn't load checkpoint {u.checkpoints_path}/{latest_step}")
        print(f"Error: {e}")
        exit_flag = True

    mu_rs = []
    c_rs = []

    for i in range(num_it):

        loss_data, d_mu_p, d_c_p = func_step(mu_p, c_p, ATT_masks, P_data, **func_step_kwargs)

        updates_mu, opt_mu_state = opt_mu.update(d_mu_p, opt_mu_state)
        d_c_p = state_r[i].apply_fn(state_r[i].params,
                                                mu_p.on_grid,
                                                d_mu_p.on_grid,
                                                c_p.on_grid, 
                                                d_c_p.on_grid, 
                                                train=False)
 
        updates_c, opt_c_state = opt_c.update(d_c_p, opt_c_state)
        
        c_p = optax.apply_updates(c_p, updates_c)
        mu_p = optax.apply_updates(mu_p, updates_mu)
        
        mu_r = get_mu(mu_p)
        c_r = get_sound_speed(c_p)

        losses["data"].append(loss_data)
        mu_rs.append(mu_r.on_grid.squeeze())
        c_rs.append(c_r.on_grid.squeeze())
        losses["c"].append(mse(c_rs[-1], c)/jnp.mean(c))
        losses["mu"].append(mse(mu_rs[-1], mu)/jnp.mean(mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_2opt(dataset, file_index, func_step, lr=[1.,1.], num_illum=np.inf, num_it=u.RECON_ITERATIONS, func_step_kwargs={}):
    """
    Calls a reconstruction function using 2 optimizers and gradient sharing. Optionally, it can be limited to a number of illuminations.

    If performing multiple reconstructions, it may be preferrable to call recon_batch, feeding this as the func_recon parameter instead.

    Args:
        dataset (PADataset): Dataset to use.
        file_index (int): File index to reconstruct.
        func_step (function): Reconstruction function.
        lr (list): Learning rates for mu and c. Default is [1., 1.].
        num_illum (int): Number of illuminations to use. Default is np.inf.
        num_it (int): Number of iterations. Default is u.RECON_ITERATIONS.
        func_step_kwargs (dict): Additional arguments for the reconstruction function.

    Returns:
        tuple: Reconstruction losses and results.
    """
    jax.clear_caches()
    num_angles = dataset.num_angles
    illum_indices = np.linspace(0, num_angles, num_illum, endpoint=False).astype(int)  if num_illum < np.inf else np.array(range(num_angles))
    data = dataset[(file_index, illum_indices)]

    print(f"illumination angles: {data["angles"]}")

    j = data["file_idx"]
    mu = data["mu"]
    ATT_masks = FourierSeries(data["ATT_masks"], domain)
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    c_p = FourierSeries(jnp.zeros(im_shape)-5, domain) 
    mu_p = FourierSeries(jnp.zeros(im_shape)+1, domain) 

    opt_mu = optax.adam(learning_rate=lr[0])
    opt_c = optax.adam(learning_rate=lr[1])
    opt_mu_state = opt_mu.init(mu_p)
    opt_c_state = opt_c.init(c_p)

    mu_rs = []
    c_rs = []

    for i in range(num_it):

        loss_data, d_mu_p, d_c_p = func_step(mu_p, c_p, ATT_masks, P_data, **func_step_kwargs)

        updates_c, opt_c_state = opt_c.update(d_mu_p, opt_c_state) # Share gradients
        updates_mu, opt_mu_state = opt_mu.update(d_mu_p, opt_mu_state)

        c_p = optax.apply_updates(c_p, updates_c)
        mu_p = optax.apply_updates(mu_p, updates_mu)
        
        mu_r = get_mu(mu_p)
        c_r = get_sound_speed(c_p)

        losses["data"].append(loss_data)
        mu_rs.append(mu_r.on_grid.squeeze())
        c_rs.append(c_r.on_grid.squeeze())
        losses["c"].append(mse(c_rs[-1], c)/jnp.mean(c))
        losses["mu"].append(mse(mu_rs[-1], mu)/jnp.mean(mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_1opt(dataset, file_index, func_step, lr=1., num_illum=np.inf, num_it=u.RECON_ITERATIONS, func_step_kwargs={}):
    """
    Calls a reconstruction function using a single optimizer and gradient sharing. Optionally, it can be limited to a number of illuminations.

    If performing multiple reconstructions, it may be preferrable to call recon_batch, feeding this as the func_recon parameter instead.

    Args:
        dataset (PADataset): Dataset to use.
        file_index (int): File index to reconstruct.
        func_step (function): Reconstruction function.
        lr (list): Learning rates for mu and c. Default is 1. Be careful not to feed this a tuple or list!
        num_illum (int): Number of illuminations to use. Default is np.inf.
        num_it (int): Number of iterations. Default is u.RECON_ITERATIONS.
        func_step_kwargs (dict): Additional arguments for the reconstruction function.

    Returns:
        tuple: Reconstruction losses and results.
    """
    jax.clear_caches()
    num_angles = dataset.num_angles
    illum_indices = np.linspace(0, num_angles, num_illum, endpoint=False).astype(int)  if num_illum < np.inf else np.array(range(num_angles))
    data = dataset[(file_index, illum_indices)]
    print(f"illumination angles: {data["angles"]}")

    j = data["file_idx"]
    mu = data["mu"]
    ATT_masks = FourierSeries(data["ATT_masks"], domain)
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    c_p = FourierSeries(jnp.zeros(im_shape)-5, domain) 
    mu_p = FourierSeries(jnp.zeros(im_shape)+1, domain) 

    fields = (mu_p, c_p)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(fields)

    mu_rs = []
    c_rs = []

    for i in range(num_it):

        loss_data, d_mu_p, d_c_p = func_step(mu_p=fields[0], c_p=fields[1], ATT_masks=ATT_masks, P_data=P_data, **func_step_kwargs)

        updates_fields, opt_state = opt.update((d_mu_p, d_mu_p), opt_state) # Gradient sharing
        
        fields = optax.apply_updates(fields, updates_fields)

        mu_r = get_mu(fields[0])
        c_r = get_sound_speed(fields[1])

        losses["data"].append(loss_data)
        mu_rs.append(mu_r.on_grid.squeeze())
        c_rs.append(c_r.on_grid.squeeze())
        losses["c"].append(mse(c_rs[-1], c)/jnp.mean(c))
        losses["mu"].append(mse(mu_rs[-1], mu)/jnp.mean(mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_batch(func_recon, func_step, file_indices=range(u.RECON_FILE_START, u.RECON_FILE_END), save=True, print_losses=True, lr=[1.,1.], num_illum=np.inf, num_it=u.RECON_ITERATIONS, func_step_kwargs={}):
    """
    Batch reconstruction function. Calls a reconstruction function for multiple files.

    Args:
        func_recon (function): Reconstruction function.
        func_step (function): Reconstruction step function.
        file_indices (list): List of file indices to reconstruct. Default is range(u.RECON_FILE_START, u.RECON_FILE_END).
        save (bool): Save the reconstructions. Default is True.
        print_losses (bool): Print the losses. Default is True.
        lr (list): Learning rates for mu and c. Default is [1., 1.].
        num_illum (int): Number of illuminations to use. Default is np.inf.
        num_it (int): Number of iterations. Default is u.RECON_ITERATIONS.
        func_step_kwargs (dict): Additional arguments for the reconstruction function.

    Returns:
        tuple: Reconstruction losses and results.
    """ 
    global exit_flag
    key = random.PRNGKey(43)
    dataset = PADataset()

    recon_batch = defaultdict(dict)
    losses_batch = defaultdict(dict)

    for file_index in file_indices:
        if exit_flag:
            break

        j = file_index
        
        losses, recon = func_recon(dataset, file_index, func_step, lr=lr, num_illum=num_illum, num_it=num_it, func_step_kwargs=func_step_kwargs)

        recon_batch[j] = recon
        losses_batch[j] = losses

        if save:
            save_recon(j, recon)
        if print_losses:
            print_recon_losses(j, losses)
    return losses_batch, recon_batch

# --------------------------------------------
def print_nets():
    """
    Print the network architectures.
    """
    print_net(R, [im_shape, im_shape, im_shape, im_shape])
    # print_net(model, [P0_shape, P0_shape, im_shape, im_shape])

exit_flag = False
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="t", nargs="?")
    parser.add_argument("-l", dest="illum", type=int, default=np.inf, nargs="?")
    parser.add_argument("-i", dest="iter", type=int, default=u.RECON_ITERATIONS, nargs="?")
    parser.add_argument("-f", dest="f", type=str, default=None, nargs="?")
    parser.add_argument("-c", dest="c", action="store_true")
    args = parser.parse_args()
    
    f = recon_step
    if args.f is not None:
        f = eval(args.f)

    if args.mode == "t":
        print(f"Continue training: {args.c}")
        train_r_c(cont=args.c)
    
    elif args.mode == "r1":
        recon_batch(func_recon=recon_1opt, func_step=recon_step, num_it=args.iter, num_illum=4, func_step_kwargs={}, lr=1.)
    
    elif args.mode == "r2":
        recon_batch(func_recon=recon_2opt, func_step=recon_step, num_it=args.iter, num_illum=args.illum, func_step_kwargs={})

    elif args.mode == "r3":
        recon_batch(func_recon=recon_2opt_r, func_step=recon_step, num_it=args.iter, num_illum=args.illum, func_step_kwargs={})
    
    elif args.mode == "p":
        print_nets()
