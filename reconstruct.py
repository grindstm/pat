import os
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
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_utils

from jwave.geometry import Domain, Medium, BLISensors, TimeAxis
from jwave import simulate_wave_propagation

from jaxdf import FourierSeries
from jaxdf.operators import compose

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
    iteration_count = len(recon["mu_rs"])
    for i in range(iteration_count):
        mu_r_file = u.file(u.mu_r_path, j, i)
        jnp.save(mu_r_file, recon["mu_rs"][i].squeeze())
        c_r_file = u.file(u.c_r_path, j, i)
        jnp.save(c_r_file, recon["c_rs"][i].squeeze())
    # jnp.save(u.state_path, state)

def print_losses(j, state, reg=True):
    print(
        "***WARNING***: Training files out of order results in incorrect loss values. Start from file 0 and don't skip. (TODO)"
    )
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
            compute_vjp_flops=True,
        )
    )

if u.DIMS == 2:
    N = u.N[:2]
    DX = u.DX[:2]

P0_shape = (u.NUM_LIGHTING_ANGLES, *N, 1)
im_shape = (1, *N, 1)

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
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
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

class ConcatNet(nn.Module):
    # concatenate all 4 fields and encode, then decode into 2 fields. Skip connections are omitted.
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

class ConcatNetSkip(nn.Module):
    # concatenate all 4 fields and encode, then decode into 2 fields. Skip connections are convolved to reduce features.
    pass

class XNetLike(nn.Module):
    # 4 fields are separately encoded, then decoded using skip connections from other relevant fields.
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x0, x1, x2, x3, train: bool = True):
        Encoder = partial(
            EncoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Conver = partial(
            ConvBlock, dropout=self.dropout, activation=self.activation, train=train
        )
        Decoder = partial(
            DecoderBlock, dropout=self.dropout, activation=self.activation, train=train
        )

        e0_0 = Encoder()(x0, self.features)
        e0_1 = Encoder()(e0_0, self.features * 2)
        e0_2 = Encoder()(e0_1, self.features * 4)

        e1_0 = Encoder()(x1, self.features)
        e1_1 = Encoder()(e1_0, self.features * 2)
        e1_2 = Encoder()(e1_1, self.features * 4)

        e2_0 = Encoder()(x2, self.features)
        e2_1 = Encoder()(e2_0, self.features * 2)
        e2_2 = Encoder()(e2_1, self.features * 4)

        e3_0 = Encoder()(x3, self.features)
        e3_1 = Encoder()(e3_0, self.features * 2)
        e3_2 = Encoder()(e3_1, self.features * 4)

        x = jnp.concatenate([e0_2, e1_2, e2_2, e3_2], axis=-1)
        x = Conver(features=self.features * 4)(x)
        x = Conver(features=self.features * 4)(x)

        #  = jnp.concatenate([
        o0 = Decoder()(x, self.features * 4)

class XNetKnit(nn.Module):
    # 4 fields are separately encoded, then decoded using skip connections from all other fields. Convolutions are used in skip connections.
    pass

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
        x0: P0_r
        x1: d_P0
        x2: c_r
        x3: d_c
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
):
    key, key_init, key_dropout, *key_input = random.split(key, 3 + len(shapes))

    inputs = (random.normal(key_input[i], shapes[i]) for i in range(len(shapes)))

    variables = model.init(
        key_init,
        *inputs,
    )

    batch_stats = variables["batch_stats"]
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        key=key_dropout,
        batch_stats=batch_stats,
        tx=tx
    )

def compute_metrics(*, state, data):
    updates=state.metrics.single_from_model_output
    metrics=state.metrics.merge(updates)
    state=state.replace(metrics=metrics)
    
# --------------------------------------------

features_0 = 32
model = StepNet(features=features_0, dropout=0.1, activation="elu", network=ConcatNet)

# --------------------------------------------

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


def train_step(j, data, state, key):
    key, key_dropout = random.split(key)
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [np.nan], "c": [], "sum_mu_c":[np.nan]}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    P0_r = jnp.zeros(P0_shape)
    c_r = jnp.ones(im_shape) * u.C

    mu_rs = [mu_r]
    c_rs = [c_r]

    P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
    losses["data"].append(mse(P_pred.squeeze(), P_data))

    P0_r = P0_r - u.LR_MU_R * d_P0
    P0_r = jnp.clip(P0_r.on_grid, 0.0)
    c_r = c_r - u.LR_C_R * d_c
    c_rs.append(c_r)
    
    losses["c"].append(mse(c_r.squeeze(), c))


    for i in range(u.RECON_ITERATIONS):
        P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))

        def loss_fn(params, batch_stats):
            (mu_r_, c_r_), updates_bs = state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                P0_r,
                d_P0.on_grid,
                c_r,
                d_c,
                train=True,
                rngs={"dropout": key_dropout},
                mutable=["batch_stats"],
            )
            loss = mse(mu_r_, mu) + mse(c_r_, c)
            return loss, (mu_r_, c_r_, updates_bs)

        v_g_loss = value_and_grad(loss_fn, has_aux=True, argnums=0)
        (loss, (mu_r, c_r, updates_bs)), gradient = v_g_loss(
            state.params, state.batch_stats
        )
        state = state.apply_gradients(grads=gradient)
        state = state.replace(batch_stats=updates_bs)

        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(loss)

        mu_rs.append(mu_r)
        c_rs.append(c_r)

        P0_r = mu_r * ATT_masks


    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    state.losses[j] = losses

    return state, recon

def train(cont=False):
    global exit_flag

    key = random.PRNGKey(42)
    key, key_train_state = random.split(key)
    state = create_train_state(
        key_train_state,
        model,
        learning_rate=u.LR_R_MU,
        shapes=[P0_shape, P0_shape, im_shape, im_shape],
    )

    dataset = PADataset()
    train_data = [dataset[i] for i in range(u.TRAIN_FILE_START, u.TRAIN_FILE_END)]

    for data in train_data:
        if exit_flag:
            break

        j = data["file_idx"]

        state, recon = train_step(j, data, state, key)

        save_recon(j, recon)

        print_losses(j, state)

# --------------------------------------------
# Reconstruction
# --------------------------------------------

def get_sound_speed(params):
    return 1420.0 + 100.0*compose(params)(nn.sigmoid)

def get_mu(params):
    return compose(params)(nn.softplus)

def recon_step_full_opt2_p(j, data, key, lr=[u.LR_MU_R, u.LR_C_R], num_it=u.RECON_ITERATIONS):
    """
    Adam optimizer for each field.
    """
    mu = data["mu"]
    ATT_masks = FourierSeries(data["ATT_masks"], domain)
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    key, key_mu_init, key_c_init = random.split(key,3)
    mu_r = random.normal(key_mu_init, im_shape)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)
    c_p = FourierSeries(c_r, domain) / u.C
    mu_p = FourierSeries(mu_r, domain) 

    opt_mu = optax.adam(learning_rate=lr[0])
    opt_c = optax.adam(learning_rate=lr[1])
    opt_mu_state = opt_mu.init(mu_p)
    opt_c_state = opt_c.init(c_p)

    mu_rs = []
    c_rs = []

    for i in range(num_it):

        def loss_rec(mu_p, c_p):
            mu_r = get_mu(mu_p)
            c_r = get_sound_speed(c_p)[0]
            P0 = mu_r * ATT_masks
            medium = Medium(domain=domain, sound_speed=c_r, pml_size=u.PML_MARGIN[0])
            P_pred = batch_compiled_simulate(medium, time_axis, P0)
            return mse(P_pred.squeeze(), P_data)
        
        loss_data, (d_mu_p, d_c_p) = value_and_grad(loss_rec, (0, 1))(mu_p, c_p)

        losses["data"].append(loss_data)

        updates_c, opt_mu_state = opt_c.update(d_c_p, opt_c_state)
        c_p = optax.apply_updates(c_p, updates_c)
        updates_mu, opt_c_state = opt_mu.update(d_mu_p, opt_mu_state)
        mu_p = optax.apply_updates(mu_p, updates_mu)
        
        mu_r = get_mu(mu_p)
        c_r = get_sound_speed(c_p)

        mu_rs.append(mu_r.on_grid.squeeze())
        c_rs.append(c_r.on_grid.squeeze())
        losses["c"].append(mse(c_rs[-1], c)/jnp.mean(c))
        losses["mu"].append(mse(mu_rs[-1], mu)/jnp.mean(mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

@jit
def A(mu_p, ATT_masks, c_p):
    mu = get_mu(mu_p)
    c = get_sound_speed(c_p)
    P0 = mu * ATT_masks
    medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN[0])
    return batch_compiled_simulate(medium, time_axis, P0)

@jit
def ATr(
    mu_p,
    ATT_masks,
    c_p,
    P_data,
):
    
    P_pred, AT = jax.vjp(A, mu_p, ATT_masks, c_p)
    residual = P_pred - jnp.expand_dims(P_data, -1)
    d_mu, d_ATT_masks, d_c = AT(residual)
    # d_c = jnp.expand_dims(d_c, 0)
    return P_pred, d_mu, d_c


@jit
def recon_step_ATr_1opt(fields, ATT_masks, P_data):
    P_pred, d_mu_p, d_c_p = ATr(fields[0], ATT_masks, fields[1], P_data)
    return mse(P_pred.squeeze(), P_data), d_mu_p, d_c_p

@jit
def recon_step_ATr_2opt(mu_p, c_p, ATT_masks, P_data):
    P_pred, d_mu_p, d_c_p = ATr(mu_p, ATT_masks, c_p[0], P_data)
    return mse(P_pred.squeeze(), P_data), d_mu_p, d_c_p
        

def reg_L2(mu_p, c_p):
    return jnp.sum(mu_p.params**2) + jnp.sum(c_p.params**2)

@jit
def recon_step_full(mu_p, c_p, ATT_masks, P_data):
    def loss_rec(mu_p, c_p):
        mu_r = get_mu(mu_p)
        c_r = get_sound_speed(c_p)[0]
        P0 = mu_r * ATT_masks
        medium = Medium(domain=domain, sound_speed=c_r, pml_size=u.PML_MARGIN[0])
        P_pred = batch_compiled_simulate(medium, time_axis, P0)
        return mse(P_pred.squeeze(), P_data)
    
    loss_data, (d_mu_p, d_c_p) = value_and_grad(loss_rec, (0, 1))(mu_p, c_p)

    return loss_data, d_mu_p, d_c_p

@jit
def recon_step_full_r(mu_p, c_p, ATT_masks, P_data, r_mu, r_c):
    def loss_rec(mu_p, c_p):
        mu_r = get_mu(mu_p)
        c_r = get_sound_speed(c_p)[0]
        P0 = mu_r * ATT_masks
        medium = Medium(domain=domain, sound_speed=c_r, pml_size=u.PML_MARGIN[0])
        P_pred = batch_compiled_simulate(medium, time_axis, P0)
        return mse(P_pred.squeeze(), P_data) + r_mu * jnp.sum(mu_p.on_grid**2) + r_c* jnp.sum(c_p.on_grid**2)
    loss_data, (d_mu_p, d_c_p) = value_and_grad(loss_rec, (0, 1))(mu_p, c_p)

    return loss_data, d_mu_p, d_c_p


def train_r(num_illum=10, func_step=recon_step_full_r, lr=[u.LR_MU_R, u.LR_C_R], num_it=u.RECON_ITERATIONS, cont=False):
    global exit_flag
    key = random.PRNGKey(59)
    dataset = PADataset()

    r_mu = 1e-9
    r_c = 1e-9
    opt_r_mu = optax.adam(learning_rate=1e-8)
    opt_r_c = optax.adam(learning_rate=1e-8)
    opt_r_mu_state = opt_r_mu.init(r_mu)
    opt_r_c_state = opt_r_c.init(r_c)


    TX = TypeVar("TX", bound=optax.OptState)

    def restore_optimizer_state(opt_state: TX, restored: Mapping[str, ...]) -> TX:
        """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
        return tree_util.tree_unflatten(
            tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
        )

    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    mngr = ocp.CheckpointManager(
        os.path.abspath(u.checkpoints_path),
        item_names={"rs", "opt_r_mu_state", "opt_r_c_state"},
        options=options)

    latest_step = mngr.latest_step()
    if cont and latest_step is not None:
        try:
            restored = mngr.restore(latest_step, args=ocp.args.Composite(
                rs = ocp.args.StandardRestore(),
                opt_r_mu_state = ocp.args.PyTreeRestore(),
                opt_r_c_state = ocp.args.PyTreeRestore()))
            r_mu=restored["rs"]["r_mu"]
            r_c = restored["rs"]["r_c"]
            opt_r_mu_state = restore_optimizer_state(opt_r_mu_state, restored["opt_r_mu_state"])
            opt_r_c_state= restore_optimizer_state(opt_r_c_state, restored["opt_r_c_state"])
            
            print(f"Restored checkpoint {u.checkpoints_path}/{latest_step}")
        except Exception as e:
            print(
                f"Couldn't load checkpoint {u.checkpoints_path}/{latest_step}\n Run again without -c flag to start from scratch"
            )
            print(f"mngr error: {e}")
            exit_flag = True
    else:
        # ocp.test_utils.erase_and_create_empty(u.checkpoints_path)
        shutil.rmtree(u.checkpoints_path)
        os.makedirs(u.checkpoints_path, exist_ok=True)
        print("Created empty checkpoint folder")

    start = latest_step if cont else u.TRAIN_FILE_START
    for file_index in range(start, u.TRAIN_FILE_END):
        if exit_flag:
            break

        illum_indices = np.linspace(0, u.NUM_LIGHTING_ANGLES-1, num_illum).astype(int) 
        # if num_illum < np.inf else np.array(range(u.NUM_LIGHTING_ANGLES))
        data = dataset[(file_index, illum_indices)]
        print(f"illumination angles: {data["angles"]}")
        
        j = file_index
        mu = data["mu"]
        ATT_masks = FourierSeries(data["ATT_masks"], domain)
        c = data["c"]
        P_data = data["P_data"]
        losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
        recon = defaultdict(dict)

        key, key_mu_init, key_c_init = random.split(key,3)
        mu_r = random.normal(key_mu_init, im_shape)
        c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)
        c_p = FourierSeries(c_r, domain) / u.C
        mu_p = FourierSeries(mu_r, domain) 

        opt_mu = optax.adam(learning_rate=lr[0])
        opt_c = optax.adam(learning_rate=lr[1])
        opt_mu_state = opt_mu.init(mu_p)
        opt_c_state = opt_c.init(c_p)

        mu_rs = []
        c_rs = []

        for i in range(num_it):
            def loss_fn(r_mu, r_c, opt_mu_state, opt_c_state):
                loss_data, d_mu_p, d_c_p = func_step(mu_p, c_p, ATT_masks, P_data, r_mu, r_c)

                updates_c, opt_mu_state = opt_c.update(d_c_p, opt_c_state)
                c_p_ = optax.apply_updates(c_p, updates_c)
                updates_mu, opt_c_state = opt_mu.update(d_mu_p, opt_mu_state)
                mu_p_ = optax.apply_updates(mu_p, updates_mu)
                
                mu_r = get_mu(mu_p_)
                c_r = get_sound_speed(c_p_)

                return mse(mu_r.on_grid, mu)/jnp.mean(mu) + mse(c_r.on_grid, c)/jnp.mean(c) + 100 * loss_data, (loss_data, mu_r, c_r)

            (loss_rec, (loss_data, mu_r, c_r)), (d_r_mu, d_r_c) = value_and_grad(loss_fn, (0, 1), has_aux=True)(r_mu, r_c, opt_mu_state, opt_c_state)

            updates_r_mu, opt_r_mu_state = opt_r_mu.update(d_r_mu, opt_r_mu_state)
            r_mu = optax.apply_updates(r_mu, updates_r_mu)
            updates_r_c, opt_r_c_state = opt_r_c.update(d_r_c, opt_r_c_state)
            r_c = optax.apply_updates(r_c, updates_r_c)
            print(f"r_mu: {r_mu:.4e}, r_c: {r_c:.4e}, loss_rec: {loss_rec:.6f}")
            
            losses["data"].append(loss_data)
            mu_rs.append(mu_r.on_grid.squeeze())
            c_rs.append(c_r.on_grid.squeeze())
            losses["c"].append(mse(c_rs[-1], c)/jnp.mean(c))
            losses["mu"].append(mse(mu_rs[-1], mu)/jnp.mean(mu))
            losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
        recon["mu_rs"] = mu_rs
        recon["c_rs"] = c_rs
        losses[j] = losses

        save_recon(j, recon)
        print_recon_losses(j, losses)

        rs = {
            "r_mu": r_mu,
            "r_c": r_c,
        }

        latest_step = mngr.latest_step()
        if latest_step is None:
            latest_step = -1

        save_args = orbax_utils.save_args_from_target(opt_r_mu_state)
        mngr.save(latest_step+1, args=ocp.args.Composite(
            rs=ocp.args.StandardSave(rs), 
            opt_r_mu_state=ocp.args.PyTreeSave(opt_r_mu_state,save_args=save_args), 
            opt_r_c_state=ocp.args.PyTreeSave(opt_r_c_state,save_args=save_args)))  
        mngr.wait_until_finished()

def train_r_l(num_illum=10, func_step=recon_step_full_r, lr=[u.LR_MU_R, u.LR_C_R], num_it=u.RECON_ITERATIONS, cont=False):
    global exit_flag
    key = random.PRNGKey(59)
    dataset = PADataset()

    log_r_mu = jnp.log(1e-8)
    log_r_c = jnp.log(1e-8)
    opt_r_mu = optax.adam(learning_rate=.1)
    opt_r_c = optax.adam(learning_rate=.1)
    opt_r_mu_state = opt_r_mu.init(log_r_mu)
    opt_r_c_state = opt_r_c.init(log_r_c)


    TX = TypeVar("TX", bound=optax.OptState)

    def restore_optimizer_state(opt_state: TX, restored: Mapping[str, ...]) -> TX:
        """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
        return tree_util.tree_unflatten(
            tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
        )

    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    mngr = ocp.CheckpointManager(
        os.path.abspath(u.checkpoints_path),
        item_names={"rs", "opt_r_mu_state", "opt_r_c_state"},
        options=options)

    latest_step = mngr.latest_step()
    if cont and latest_step is not None:
        try:
            restored = mngr.restore(latest_step, args=ocp.args.Composite(
                rs = ocp.args.StandardRestore(),
                opt_r_mu_state = ocp.args.PyTreeRestore(),
                opt_r_c_state = ocp.args.PyTreeRestore()))
            log_r_mu=restored["rs"]["r_mu"]
            log_r_c = restored["rs"]["r_c"]
            opt_r_mu_state = restore_optimizer_state(opt_r_mu_state, restored["opt_r_mu_state"])
            opt_r_c_state= restore_optimizer_state(opt_r_c_state, restored["opt_r_c_state"])
            
            print(f"Restored checkpoint {u.checkpoints_path}/{latest_step}")
        except Exception as e:
            print(
                f"Couldn't load checkpoint {u.checkpoints_path}/{latest_step}\n Run again without -c flag to start from scratch"
            )
            print(f"mngr error: {e}")
            exit_flag = True
    else:
        # ocp.test_utils.erase_and_create_empty(u.checkpoints_path)
        shutil.rmtree(u.checkpoints_path)
        os.makedirs(u.checkpoints_path, exist_ok=True)
        print("Created empty checkpoint folder")

    start = latest_step if cont else u.TRAIN_FILE_START
    for file_index in range(start, u.TRAIN_FILE_END):
        if exit_flag:
            break

        illum_indices = np.linspace(0, u.NUM_LIGHTING_ANGLES-1, num_illum).astype(int) 
        # if num_illum < np.inf else np.array(range(u.NUM_LIGHTING_ANGLES))
        data = dataset[(file_index, illum_indices)]
        print(f"illumination angles: {data["angles"]}")
        
        j = file_index
        mu = data["mu"]
        ATT_masks = FourierSeries(data["ATT_masks"], domain)
        c = data["c"]
        P_data = data["P_data"]
        losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
        recon = defaultdict(dict)

        key, key_mu_init, key_c_init = random.split(key,3)
        mu_r = random.normal(key_mu_init, im_shape)
        c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)
        c_p = FourierSeries(c_r, domain) / u.C
        mu_p = FourierSeries(mu_r, domain) 

        opt_mu = optax.adam(learning_rate=lr[0])
        opt_c = optax.adam(learning_rate=lr[1])
        opt_mu_state = opt_mu.init(mu_p)
        opt_c_state = opt_c.init(c_p)

        mu_rs = []
        c_rs = []

        for i in range(num_it):
            def loss_fn(log_r_mu, log_r_c, opt_mu_state, opt_c_state, mu_p, c_p):
                r_mu = jnp.exp(log_r_mu)
                r_c = jnp.exp(log_r_c)
                loss_data, d_mu_p, d_c_p = func_step(mu_p, c_p, ATT_masks, P_data, r_mu, r_c)

                updates_c, opt_mu_state = opt_c.update(d_c_p, opt_c_state)
                c_p = optax.apply_updates(c_p, updates_c)
                updates_mu, opt_c_state = opt_mu.update(d_mu_p, opt_mu_state)
                mu_p = optax.apply_updates(mu_p, updates_mu)
                
                mu_r = get_mu(mu_p)
                c_r = get_sound_speed(c_p)

                return mse(mu_r.on_grid, mu)/jnp.mean(mu) + mse(c_r.on_grid, c)/jnp.mean(c) + loss_data / jnp.mean(P_data), (loss_data, mu_r, c_r, mu_p, c_p)

            (loss_rec, (loss_data, mu_r, c_r, mu_p, c_p)), (d_r_mu, d_r_c) = value_and_grad(loss_fn, (0, 1), has_aux=True)(log_r_mu, log_r_c, opt_mu_state, opt_c_state, mu_p, c_p)

            updates_r_mu, opt_r_mu_state = opt_r_mu.update(d_r_mu, opt_r_mu_state)
            log_r_mu = optax.apply_updates(log_r_mu, updates_r_mu)
            updates_r_c, opt_r_c_state = opt_r_c.update(d_r_c, opt_r_c_state)
            log_r_c = optax.apply_updates(log_r_c, updates_r_c)
            print(f"r_mu: {jnp.exp(log_r_mu):.4e}, r_c: {jnp.exp(log_r_c):.4e}, loss_rec: {loss_rec:.6f}")
            
            losses["data"].append(loss_data)
            mu_rs.append(mu_r.on_grid.squeeze())
            c_rs.append(c_r.on_grid.squeeze())
            losses["c"].append(mse(c_rs[-1], c)/jnp.mean(c))
            losses["mu"].append(mse(mu_rs[-1], mu)/jnp.mean(mu))
            losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
        recon["mu_rs"] = mu_rs
        recon["c_rs"] = c_rs
        losses[j] = losses

        save_recon(j, recon)
        print_recon_losses(j, losses)

        rs = {
            "r_mu": log_r_mu,
            "r_c": log_r_c,
        }

        latest_step = mngr.latest_step()
        if latest_step is None:
            latest_step = -1

        save_args = orbax_utils.save_args_from_target(opt_r_mu_state)
        mngr.save(latest_step+1, args=ocp.args.Composite(
            rs=ocp.args.StandardSave(rs), 
            opt_r_mu_state=ocp.args.PyTreeSave(opt_r_mu_state,save_args=save_args), 
            opt_r_c_state=ocp.args.PyTreeSave(opt_r_c_state,save_args=save_args)))  
        mngr.wait_until_finished()

def recon_batch(func_recon, func_step, file_indices=range(u.RECON_FILE_START, u.RECON_FILE_END), save=True, print_losses=True, lr=[1.,1.], num_illum=np.inf, num_it=u.RECON_ITERATIONS, func_step_kwargs={}):
    global exit_flag
    key = random.PRNGKey(43)
    dataset = PADataset()

    recon_batch = defaultdict(dict)
    losses_batch = defaultdict(dict)

    for file_index in file_indices:
        if exit_flag:
            break

        j = file_index
        
        losses, recon = func_recon(key, dataset, file_index, func_step, lr=lr, num_illum=num_illum, num_it=num_it, func_step_kwargs=func_step_kwargs)

        recon_batch[j] = recon
        losses_batch[j] = losses

        if save:
            save_recon(j, recon)
        if print_losses:
            print_recon_losses(j, losses)
    return losses_batch, recon_batch

def recon_2opt_r(key, dataset, file_index, func_step, lr=[1.,1.], num_illum=np.inf, num_it=u.RECON_ITERATIONS, func_step_kwargs={}):
    """
    Calls a reconstruction function for each file in the reconstruction dataset. Optionally, it can be limited to a number of illuminations.
    Saves the reconstructions and prints the losses.
    """
    jax.clear_caches()
    num_angles = dataset.num_angles
    illum_indices = np.linspace(0, num_angles-1, num_illum).astype(int)  if num_illum < np.inf else np.array(range(num_angles))
    data = dataset[(file_index, illum_indices)]
    # illum_indices = np.linspace(0, len(data[0]['angles'])-1, num_illum).astype(int)  if num_illum < np.inf else np.array(range(len(data[0]['angles'])))

    print(f"illumination angles: {data["angles"]}")

    j = data["file_idx"]
    mu = data["mu"]
    ATT_masks = FourierSeries(data["ATT_masks"], domain)
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    key, key_mu_init, key_c_init = random.split(key,3)
    mu_r = random.normal(key_mu_init, im_shape)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)
    c_p = FourierSeries(c_r, domain) / u.C
    mu_p = FourierSeries(mu_r, domain) 

    opt_mu = optax.adam(learning_rate=lr[0])
    opt_c = optax.adam(learning_rate=lr[1])
    opt_mu_state = opt_mu.init(mu_p)
    opt_c_state = opt_c.init(c_p)

    mu_rs = []
    c_rs = []

    for i in range(num_it):

        loss_data, d_mu_p, d_c_p = func_step(mu_p, c_p, ATT_masks, P_data, **func_step_kwargs)

        updates_c, opt_mu_state = opt_c.update(d_c_p, opt_c_state)
        c_p = optax.apply_updates(c_p, updates_c)
        updates_mu, opt_c_state = opt_mu.update(d_mu_p, opt_mu_state)
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

def recon_2opt(func_step, lr=[1.,1.], num_illum=np.inf, num_it=u.RECON_ITERATIONS, func_step_kwargs={}):
    """
    Calls a reconstruction function for each file in the reconstruction dataset. Optionally, it can be limited to a number of illuminations.
    Saves the reconstructions and prints the losses.
    """
    global exit_flag
    key = random.PRNGKey(43)
    dataset = PADataset()

    illum_indices = np.linspace(0, len(dataset[0]['angles'])-1, num_illum).astype(int)  if num_illum < np.inf else np.array(range(len(dataset[0]['angles'])))

    recon_data = [dataset[(i, illum_indices)] for i in range(u.RECON_FILE_START, u.RECON_FILE_END)]

    for data in recon_data:
        if exit_flag:
            break

        print(f"illumination angles: {data["angles"]}")

        j = data["file_idx"]
        mu = data["mu"]
        ATT_masks = FourierSeries(data["ATT_masks"], domain)
        c = data["c"]
        P_data = data["P_data"]

        losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
        recon = defaultdict(dict)

        key, key_mu_init, key_c_init = random.split(key,3)
        mu_r = random.normal(key_mu_init, im_shape)
        c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)
        c_p = FourierSeries(c_r, domain) / u.C
        mu_p = FourierSeries(mu_r, domain) 

        opt_mu = optax.adam(learning_rate=lr[0])
        opt_c = optax.adam(learning_rate=lr[1])
        opt_mu_state = opt_mu.init(mu_p)
        opt_c_state = opt_c.init(c_p)

        mu_rs = []
        c_rs = []

        for i in range(num_it):

            loss_data, d_mu_p, d_c_p = func_step(mu_p, c_p, ATT_masks, P_data, **func_step_kwargs)

            updates_c, opt_mu_state = opt_c.update(d_c_p, opt_c_state)
            c_p = optax.apply_updates(c_p, updates_c)
            updates_mu, opt_c_state = opt_mu.update(d_mu_p, opt_mu_state)
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


        save_recon(j, recon)

        print_recon_losses(j, losses)

    return losses, recon

def recon_1opt(func_recon, lr=1., num_illum=np.inf, num_it=u.RECON_ITERATIONS):
    """
    Calls a reconstruction function for each file in the reconstruction dataset. Optionally, it can be limited to a number of illuminations.
    Saves the reconstructions and prints the losses.
    """
    global exit_flag
    key = random.PRNGKey(43)
    dataset = PADataset()

    illum_indices = np.linspace(0, len(dataset[0]['angles'])-1, num_illum).astype(int)  if num_illum < np.inf else np.array(range(len(dataset[0]['angles'])))

    recon_data = [dataset[(i, illum_indices)] for i in range(u.RECON_FILE_START, u.RECON_FILE_END)]

    for data in recon_data:
        if exit_flag:
            break

        # print(f"illumination angles: {data["angles"]}")

        j = data["file_idx"]
        mu = data["mu"]
        ATT_masks = FourierSeries(data["ATT_masks"], domain)
        c = data["c"]
        P_data = data["P_data"]

        losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
        recon = defaultdict(dict)

        key, key_mu_init, key_c_init = random.split(key,3)
        mu_r = random.normal(key_mu_init, im_shape)
        c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)
        c_p = FourierSeries(c_r[0], domain) / u.C
        mu_p = FourierSeries(mu_r, domain) 


        fields = [mu_p, c_p]
        opt = optax.adam(learning_rate=lr, b1=.9, b2=.9)
        opt_state = opt.init(fields)

        mu_rs = []
        c_rs = []

        for i in range(num_it):

            loss_data, d_mu_p, d_c_p = func_recon(fields, ATT_masks, P_data)

            updates_fields, opt_state = opt.update([d_mu_p, d_c_p], opt_state)
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

        save_recon(j, recon)

        print_recon_losses(j, losses)

    return losses, recon



# --------------------------------------------
def print_nets():
    print_net(model, [P0_shape, P0_shape, im_shape, im_shape])

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
    
    if args.mode == "t":
        print(f"Continue training: {args.c}")
        train_r_l(cont=args.c)
        # train_r(cont=args.c)
        # train(args.c)
    
    elif args.mode == "r1":
        f = recon_step_ATr_1opt
        if args.f is not None:
            f = eval(args.f)
        recon_1opt(func_recon=f, num_illum=args.illum, num_it=args.iter)
    
    elif args.mode == "r2":
        # f = recon_step_ATr_2opt
        f = recon_step_full
        if args.f is not None:
            f = eval(args.f)
        recon_batch(func_recon=recon_2opt_r, func_step=recon_step_full_r, num_it=args.iter, num_illum=args.illum, func_step_kwargs={"r_mu":0, "r_c":0})#=args.illum)
    
    elif args.mode == "r3":
        f = recon_step_full_r
        if args.f is not None:
            f = eval(args.f)
        # recon_batch(func_recon=recon_2opt_r, func_step=recon_step_full_r, num_it=args.iter, func_step_kwargs={"r_mu":1.2710e-11, "r_c":0}, num_illum=10)#=args.illum)
        recon_batch(func_recon=recon_2opt_r, func_step=recon_step_full_r, num_it=args.iter, func_step_kwargs={"r_mu":1.2710e-11, "r_c":4.3965e-08}, num_illum=10)#=args.illum)
    elif args.mode == "p":
        print_nets()
