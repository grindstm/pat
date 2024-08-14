import os
import signal
from collections import defaultdict
import argparse
from typing import Any
from functools import partial
import numpy as np
import jax
from jax import random
from jax import vmap, jit, value_and_grad
import jax.numpy as jnp
from flax import linen as nn
from flax import struct 

# from flax import serialization
from flax.training import train_state
from flax.training import orbax_utils
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_utils
from jwave.geometry import Domain, Medium, BLISensors, TimeAxis
from jwave import simulate_wave_propagation
from jaxdf import FourierSeries
from jaxdf.operators import compose
from tqdm import tqdm
import util as u
from PADataset import PADataset
from clu import metrics

jax.clear_caches()


# --------------------------------------------
# Utilities and constants
# --------------------------------------------
def signal_handler(signum, frame):
    global exit_flag
    exit_flag = True
    print("Exit signal received, finishing current task...")


def save_recon(j, recon):
    for i in range(u.RECON_ITERATIONS):
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


@jit
def A(P0, c):
    medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN[0])
    return batch_compiled_simulate(medium, time_axis, P0)


@jit
def ATr(
    P0_r,
    c_r,
    P_data,
):
    P0_r = FourierSeries(P0_r, domain)
    c_r = FourierSeries(c_r, domain)
    P_pred, AT = jax.vjp(A, P0_r, c_r)
    residual = P_pred - jnp.expand_dims(P_data, -1)
    d_P0, d_c = AT(residual)
    d_c = jnp.expand_dims(d_c.on_grid, 0)
    return P_pred, d_P0.on_grid, d_c


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

@jit
def A_mu(mu, ATT_masks, c):
    P0 = FourierSeries(mu * ATT_masks, domain)
    c = FourierSeries(c, domain)
    medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN[0])
    return batch_compiled_simulate(medium, time_axis, P0)

@jit
def ATr_mu(
    mu_r,
    ATT_masks,
    c_r,
    P_data,
):
    
    P_pred, AT = jax.vjp(A_mu, mu_r, ATT_masks, c_r)
    residual = P_pred - jnp.expand_dims(P_data, -1)
    d_mu, d_ATT_masks, d_c = AT(residual)
    d_c = jnp.expand_dims(d_c, 0)
    return P_pred, d_mu, d_c

def mu_from_P0_mean(P0):
    return jnp.mean(P0, axis=0)

def mu_from_P0_vjp(mu_r, ATT_masks, P0_r):
    def mul(x, y):
        return x * y
    P0_, mulT = jax.vjp(mul, mu_r, ATT_masks)
    mu_r, ATT_masks_  = mulT(P0_r)
    return mu_r, ATT_masks_

def recon_step_ATrmu(j, data, key, lr=[u.LR_MU_R], num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)

    mu_rs = []
    c_rs = []

    for i in range(num_it):
        P_pred, d_mu, d_c = ATr_mu(mu_r, ATT_masks, c_r[0], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))
        mu_r = mu_r - lr[0] * d_mu
        mu_r = jnp.clip(mu_r, 0.0)
        c_r = c_r - lr[1] * d_c
        mu_rs.append(mu_r.squeeze())
        c_rs.append(c_r.squeeze())
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_step_ATrmu_opt1(j, data, key, lr=[u.LR_MU_R], num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)

    fields = [mu_r, c_r]
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(fields)

    mu_rs = []
    c_rs = []

    for i in range(num_it):
        P_pred, d_mu, d_c = ATr_mu(mu_r, ATT_masks, c_r[0], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))
        updates_fields, opt_state = opt.update([d_mu, d_c], opt_state)
        fields = optax.apply_updates(fields, updates_fields)
        mu_r, c_r = fields
        mu_r = jnp.clip(mu_r, 0.0)
        mu_rs.append(mu_r.squeeze())
        c_rs.append(c_r.squeeze())
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def get_sound_speed(params):
    return 1420.0 + 100.0*compose(params)(nn.sigmoid)

def get_mu(params):
    return compose(params)(nn.softplus)

@jit
def A_mu_p(mu_p, ATT_masks, c_p):
    mu = get_mu(mu_p)
    c = get_sound_speed(c_p)[0]
    P0 = mu * ATT_masks
    medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN[0])
    return batch_compiled_simulate(medium, time_axis, P0)
@jit
def ATr_mu_p(
    mu_p,
    ATT_masks,
    c_p,
    P_data,
):
    
    P_pred, AT = jax.vjp(A_mu_p, mu_p, ATT_masks, c_p)
    residual = P_pred - jnp.expand_dims(P_data, -1)
    d_mu, d_ATT_masks, d_c = AT(residual)
    # d_c = jnp.expand_dims(d_c, 0)
    return P_pred, d_mu, d_c

def recon_step_ATrmu_p_opt1_p(j, data, key, lr=u.LR_MU_R, num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
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
    c_p = FourierSeries(c_r[0], domain) / u.C
    mu_p = FourierSeries(mu_r, domain) 


    fields = [mu_p, c_p]
    opt = optax.adam(learning_rate=lr, b1=.9, b2=.9)
    opt_state = opt.init(fields)

    mu_rs = []
    c_rs = []

    for i in range(num_it):
        P_pred, d_mu_p, d_c_p = ATr_mu_p(fields[0], ATT_masks, fields[1], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))

        updates_fields, opt_state = opt.update([d_mu_p, d_c_p], opt_state)
        fields = optax.apply_updates(fields, updates_fields)
        mu_r = get_mu(fields[0])
        c_r = get_sound_speed(fields[1])
        # mu_r = jnp.clip(mu_r, 0.0)

        mu_rs.append(mu_r.on_grid.squeeze())
        c_rs.append(c_r.on_grid.squeeze())
        losses["c"].append(mse(c_rs[-1], c)/jnp.mean(c))
        losses["mu"].append(mse(mu_rs[-1], mu)/jnp.mean(mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_step_ATrmu_p_opt2_p(j, data, key, lr=[u.LR_MU_R, u.LR_C_R], num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
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
        P_pred, d_mu_p, d_c_p = ATr_mu_p(mu_p, ATT_masks, c_p, P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))

        updates_c, opt_mu_state = opt_c.update(d_c_p, opt_c_state)
        c_p = optax.apply_updates(c_p, updates_c)
        updates_mu, opt_c_state = opt_mu.update(d_mu_p, opt_mu_state)
        mu_p = optax.apply_updates(mu_p, updates_mu)
        
        mu_r = get_mu(mu_p)
        c_r = get_sound_speed(c_p)
        # mu_r = jnp.clip(mu_r, 0.0)

        mu_rs.append(mu_r.on_grid.squeeze())
        c_rs.append(c_r.on_grid.squeeze())
        losses["c"].append(mse(c_rs[-1], c)/jnp.mean(c))
        losses["mu"].append(mse(mu_rs[-1], mu)/jnp.mean(mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_step_ATrmu_opt2(j, data, key, lr=[u.LR_MU_R], num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    key, key_c_init, key_mu_init = random.split(key, 3)
    mu_r = jnp.ones(im_shape)+ 1.*random.normal(key_mu_init, im_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)

    opt_mu = optax.adam(learning_rate=lr[0])
    opt_c = optax.adam(learning_rate=lr[1])
    opt_mu_state = opt_mu.init(mu_r)
    opt_c_state = opt_c.init(c_r)

    mu_rs = []
    mu_rs = [] 
    c_rs = []

    for i in range(num_it):
        P_pred, d_mu, d_c = ATr_mu(mu_r, ATT_masks, c_r[0], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))
        updates_c, opt_mu_state = opt_c.update(d_c, opt_c_state)
        c_r = optax.apply_updates(c_r, updates_c)
        updates_mu, opt_c_state = opt_mu.update(d_mu, opt_mu_state)
        mu_r = optax.apply_updates(mu_r, updates_mu)
        mu_r = jnp.clip(mu_r, 0.0)
        mu_rs.append(mu_r.squeeze())
        c_rs.append(c_r.squeeze())
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_step_ATrmu_opt2_norm(j, data, key, lr=[u.LR_MU_R, u.LR_C_R], num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    key, key_c_init, key_mu_init = random.split(key, 3)
    mu_r = jnp.ones(im_shape)+ 1.*random.normal(key_mu_init, im_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)

    mu_mean, mu_std = jnp.mean(mu), jnp.std(mu)
    c_mean, c_std = jnp.mean(c), jnp.std(c)
    mu_r_n = (mu_r - mu_mean) / mu_std
    c_r_n = (c_r - c_mean) / c_std

    opt_mu = optax.adam(learning_rate=lr[0])
    opt_c = optax.adam(learning_rate=lr[1])
    opt_mu_state = opt_mu.init(mu_r_n)
    opt_c_state = opt_c.init(c_r_n)

    mu_rs = []
    mu_rs = [] 
    c_rs = []

    for i in range(num_it):
        P_pred, d_mu, d_c = ATr_mu(mu_r, ATT_masks, c_r[0], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))

        d_mu_n = d_mu / mu_std
        d_c_n =  d_c / c_std
        
        updates_mu, opt_c_state = opt_mu.update(d_mu_n, opt_mu_state)
        mu_r_n = optax.apply_updates(mu_r_n, updates_mu)
        updates_c, opt_mu_state = opt_c.update(d_c_n, opt_c_state)
        c_r_n = optax.apply_updates(c_r_n, updates_c)

        mu_r = mu_r_n * mu_std + mu_mean
        mu_r = jnp.clip(mu_r, 0.0)
        c_r = c_r_n * c_std + c_mean

        mu_rs.append(mu_r.squeeze())
        c_rs.append(c_r.squeeze())
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_step_ATrmu_opt1_norm(j, data, key, lr=u.LR_MU_R, num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    # mu_r = jnp.zeros(im_shape)
    key, key_c_init, key_mu_init = random.split(key, 3)
    mu_r = jnp.ones(im_shape)+ 1.*random.normal(key_mu_init, im_shape)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)

    mu_mean, mu_std = jnp.mean(mu), jnp.std(mu)
    c_mean, c_std = jnp.mean(c), jnp.std(c)
    mu_r_n = (mu_r - mu_mean) / mu_std
    c_r_n = (c_r - c_mean) / c_std
    fields = (mu_r_n, c_r_n)

    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(fields)

    mu_rs = []
    c_rs = []

    for i in range(num_it):
        P_pred, d_mu, d_c = ATr_mu(mu_r, ATT_masks, c_r[0], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))

        d_mu_n = d_mu / mu_std
        d_c_n =  d_c / c_std

        updates_fields, opt_state = opt.update((d_mu_n, d_c_n), opt_state)
        fields = optax.apply_updates(fields, updates_fields)

        mu_r = fields[0] * mu_std + mu_mean
        mu_r = jnp.clip(mu_r, 0.0)
        c_r = fields[1] * c_std + c_mean

        mu_rs.append(mu_r.squeeze())
        c_rs.append(c_r.squeeze())
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_step_ATrP0(j, data, key, lr=[u.LR_MU_R], num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    P0_r = jnp.zeros(P0_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)

    mu_rs = []
    c_rs = []

    for i in range(num_it):
        P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))
        P0_r = P0_r - lr[0] * d_P0
        P0_r = jnp.clip(P0_r, 0.0)
        mu_r = mu_from_P0_mean(P0_r)
        # mu_r, ATT_masks_ = mu_from_P0_vjp(mu_r, ATT_masks, P0_r)
        c_r = c_r - lr[1] * d_c

        mu_rs.append(mu_r.squeeze())
        c_rs.append(c_r.squeeze())
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses
    return losses, recon

def recon_step_ATrP0_opt1(j, data, key, lr=[u.LR_MU_R], num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    P0_r = jnp.zeros(P0_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)

    fields = [P0_r, c_r]
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(fields)

    mu_rs = []
    c_rs = []

    for i in range(num_it):
        P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))
        updates_fields, opt_state = opt.update([d_P0, d_c], opt_state)
        fields = optax.apply_updates(fields, updates_fields)
        P0_r, c_r = fields
        P0_r = jnp.clip(P0_r, 0.0)
        mu_r = mu_from_P0_mean(P0_r)
        mu_rs.append(mu_r.squeeze())
        c_rs.append(c_r.squeeze())
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_step_ATrP0_opt2(j, data, key, lr=[u.LR_MU_R], num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c":[]}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    P0_r = jnp.zeros(P0_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.*random.normal(key_c_init, im_shape)

    opt_c = optax.adam(learning_rate=lr[0])
    opt_P0 = optax.adam(learning_rate=lr[1])
    opt_P0_state = opt_P0.init(P0_r)
    opt_c_state = opt_c.init(c_r)

    mu_rs = []
    P0_rs = [] 
    c_rs = []

    for i in range(num_it):
        P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
        losses["data"].append(mse(P_pred.squeeze(), P_data))
        updates_c, opt_state = opt_c.update(d_c, opt_c_state)
        c_r = optax.apply_updates(c_r, updates_c)
        updates_P0, opt_state = opt_P0.update(d_P0, opt_P0_state)
        P0_r = optax.apply_updates(P0_r, updates_P0)

        P0_r = jnp.clip(P0_r, 0.0)
        # mu_r = mu_from_P0_mean(P0_r)
        mu_r, ATT_masks_ = mu_from_P0_vjp(mu_r, ATT_masks, P0_r)
        mu_rs.append(mu_r.squeeze())
        c_rs.append(c_r.squeeze())
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_step_backpropP0_opt1(j, data, key):
    """
    loss, (d_P0, d_c) = value_and_grad_loss(P0_r, c_r)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [np.nan], "c": [], "sum_mu_c":[np.nan]}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    P0_r = jnp.zeros(P0_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C+1.*random.normal(key_c_init, im_shape)

    # fields = [mu_r, c_r]
    # --- 4
    fields = [P0_r, c_r]
    opt = optax.adam(learning_rate=u.LR_MU_R)
    opt_state = opt.init(fields)
    # --- 1
    # opt_P0_state = opt_P0.init(P0_r)
    # opt_c_state = opt_c.init(c_r)
    # opt_c = optax.adam(learning_rate=u.LR_C_R)
    # opt_P0 = optax.adam(learning_rate=u.LR_MU_R)


    mu_rs = [mu_r]
    P0_rs = [P0_r] 
    c_rs = [c_r]

    for i in range(u.RECON_ITERATIONS):

        # P0_r = FourierSeries(P0_r, domain)
        # c_r = FourierSeries(c_r, domain)

        # @jit
        # def loss_fn(P0_r, c_r):
        #     P_pred = A(P0_r, c_r[0])
        #     return mse(P_data, P_pred.squeeze())
        # value_and_grad_loss = value_and_grad(loss_fn, argnums=(0,1))
        # loss, (d_P0, d_c) = value_and_grad_loss(P0_r, c_r)
        # --- 1
        P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
        # --- 0
        # P_pred, d_mu, d_c = ATr_mu(mu_r, ATT_masks, c_r[0], P_data)
        
        # print(jnp.max(d_P0), jnp.max(d_c))

        # losses["data"].append(loss)
        # --- 0 1
        losses["data"].append(mse(P_pred.squeeze(), P_data))

        # --- 4
        # updates_fields, opt_state = opt.update([d_mu, d_c], opt_state)
        # fields = optax.apply_updates(fields, updates_fields)
        # mu_r, c_r = fields
        # --- 3
        updates_fields, opt_state = opt.update([d_P0, d_c], opt_state)
        # updates_fields, opt_state = opt.update([d_P0.params, d_c.params], opt_state)
        fields = optax.apply_updates(fields, updates_fields)
        P0_r, c_r = fields
        # --- 1 2
        # updates_P0_r, opt_P0_state = opt_P0.update(d_P0, opt_P0_state)
        # P0_r = optax.apply_updates(P0_r, updates_P0_r)
        # --- 0 1
        # P0_r = P0_r - u.LR_MU_R * d_P0
        # --- 0 1
        # mu_r = mu_r - u.LR_MU_R * d_mu

        P0_r = jnp.clip(P0_r, 0.0)
        # mu_r = jnp.clip(mu_r, 0.0)

        mu_r = jnp.mean(P0_r, axis=0)
        # --- 2 
        # def mul(x, y):
        #     return x * y
        # P0_, mulT = jax.vjp(mul, mu_r, ATT_masks)
        # mu_r, ATT_masks_  = mulT(P0_r)

        # --- 1
        # updates_c_r, opt_c_state = opt_c.update(d_c, opt_c_state)
        # c_r = optax.apply_updates(c_r, updates_c_r)
        # --- 3
        # c_r = c_r.params - u.LR_C_R * d_c
        # --- 0
        # c_r = c_r - u.LR_C_R * d_c

        mu_rs.append(mu_r)
        c_rs.append(c_r)

        
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])

        mu_rs.append(mu_r)
        c_rs.append(c_r)

    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon

def recon_step_(j, data, key):
    """
    ATr = 
    Updates done with two adam optimizers.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [np.nan], "c": [], "sum_mu_c":[np.nan]}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    P0_r = jnp.zeros(P0_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C+1.*random.normal(key_c_init, im_shape)

    # fields = [mu_r, c_r]
    # --- 4
    fields = [P0_r, c_r]
    opt = optax.adam(learning_rate=u.LR_MU_R)
    opt_state = opt.init(fields)
    # --- 1
    # opt_P0_state = opt_P0.init(P0_r)
    # opt_c_state = opt_c.init(c_r)
    # opt_c = optax.adam(learning_rate=u.LR_C_R)
    # opt_P0 = optax.adam(learning_rate=u.LR_MU_R)


    mu_rs = [mu_r]
    P0_rs = [P0_r] 
    c_rs = [c_r]

    for i in range(u.RECON_ITERATIONS):

        # P0_r = FourierSeries(P0_r, domain)
        # c_r = FourierSeries(c_r, domain)

        # @jit
        # def loss_fn(P0_r, c_r):
        #     P_pred = A(P0_r, c_r[0])
        #     return mse(P_data, P_pred.squeeze())
        # value_and_grad_loss = value_and_grad(loss_fn, argnums=(0,1))
        # loss, (d_P0, d_c) = value_and_grad_loss(P0_r, c_r)
        # --- 1
        P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
        # --- 0
        # P_pred, d_mu, d_c = ATr_mu(mu_r, ATT_masks, c_r[0], P_data)
        
        # print(jnp.max(d_P0), jnp.max(d_c))

        # losses["data"].append(loss)
        # --- 0 1
        losses["data"].append(mse(P_pred.squeeze(), P_data))

        # --- 4
        # updates_fields, opt_state = opt.update([d_mu, d_c], opt_state)
        # fields = optax.apply_updates(fields, updates_fields)
        # mu_r, c_r = fields
        # --- 3
        updates_fields, opt_state = opt.update([d_P0, d_c], opt_state)
        # updates_fields, opt_state = opt.update([d_P0.params, d_c.params], opt_state)
        fields = optax.apply_updates(fields, updates_fields)
        P0_r, c_r = fields
        # --- 1 2
        # updates_P0_r, opt_P0_state = opt_P0.update(d_P0, opt_P0_state)
        # P0_r = optax.apply_updates(P0_r, updates_P0_r)
        # --- 0 1
        # P0_r = P0_r - u.LR_MU_R * d_P0
        # --- 0 1
        # mu_r = mu_r - u.LR_MU_R * d_mu

        P0_r = jnp.clip(P0_r, 0.0)
        # mu_r = jnp.clip(mu_r, 0.0)

        mu_r = jnp.mean(P0_r, axis=0)
        # --- 2 
        # def mul(x, y):
        #     return x * y
        # P0_, mulT = jax.vjp(mul, mu_r, ATT_masks)
        # mu_r, ATT_masks_  = mulT(P0_r)

        # --- 1
        # updates_c_r, opt_c_state = opt_c.update(d_c, opt_c_state)
        # c_r = optax.apply_updates(c_r, updates_c_r)
        # --- 3
        # c_r = c_r.params - u.LR_C_R * d_c
        # --- 0
        # c_r = c_r - u.LR_C_R * d_c

        mu_rs.append(mu_r)
        c_rs.append(c_r)

        
        losses["c"].append(mse(c_r.squeeze(), c))
        losses["mu"].append(mse(mu_r.squeeze(), mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])

        mu_rs.append(mu_r)
        c_rs.append(c_r)

    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon


def recon(func, **kwargs):
    global exit_flag
    key = random.PRNGKey(43)
    dataset = PADataset()
    recon_data = [dataset[i] for i in range(u.RECON_FILE_START, u.RECON_FILE_END)]
    for data in recon_data:
        if exit_flag:
            break

        j = data["file_idx"]

        losses, recon = func(j, data, key, **kwargs)

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
    parser.add_argument("-f", dest="f", type=str, default="recon_step_ATrmu_p_opt2_p", nargs="?")
    parser.add_argument("-c", dest="c", action="store_true")
    args = parser.parse_args()
    if args.mode == "t":
        print(f"Continue training: {args.c}")
        train(args.c)
    elif args.mode == "r":
        recon(eval(args.f))
    elif args.mode == "p":
        print_nets()
