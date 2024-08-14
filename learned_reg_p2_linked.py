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

# from flax import serialization
from flax.training import train_state
from flax.training import orbax_utils
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_utils
from jwave.geometry import Domain, Medium, BLISensors, TimeAxis
from jwave import simulate_wave_propagation
from jaxdf import FourierSeries
from tqdm import tqdm
import util as u
from PADataset import PADataset

jax.clear_caches()


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
    print(f"File {j}\nIter\tLoss_P_data\tLoss_R_mu\tLoss_R_c")

    for i in range(u.RECON_ITERATIONS):
        print(
            f"{i}\t{state['loss_P_datas'][j][i]:.6f}\t{state['state_R_mu'].loss[j*u.RECON_ITERATIONS + i]:.6f}\t{state['state_R_c'].loss[j*u.RECON_ITERATIONS + i]:.6f}"
        )


# --------------------------------------------


class EncoderBlock(nn.Module):
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        return x


class DecoderBlock(nn.Module):
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = jax.image.resize(
            x,
            (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]),
            method="bilinear",
        )

        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return x


class SharedLink(nn.Module):
    dropout: float
    features: int 
    activation: str = "relu"

    @nn.compact
    def __call__(self, x1, x2, train: bool = True):
        x = jnp.concatenate([x1, x2], axis=-1)

        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        return x
    
class DualEncode(nn.Module):
    """
    x1: x
    x2: A^T(Ax-y)
    """
    dropout: float
    shared_link: nn.Module
    activation: str = "relu"

    @nn.compact
    def __call__(self, x1, x2, train: bool = True,):
        # Encoder I
        e0 = EncoderBlock(
            features=32, dropout=self.dropout, activation=self.activation
        )(x1, train)
        e1 = EncoderBlock(
            features=64, dropout=self.dropout, activation=self.activation
        )(e0, train)

        # Encoder II
        f0 = EncoderBlock(
            features=32, dropout=self.dropout, activation=self.activation
        )(x2, train)
        f1 = EncoderBlock(
            features=64, dropout=self.dropout, activation=self.activation
        )(f0, train)

        # Merge features
        m0 = jnp.concatenate([e1, f1], axis=-1)
        m1 = nn.Conv(features=32, kernel_size=(3, 3))(m0)
        m1 = nn.BatchNorm(use_running_average=not train)(m1)
        m1 = getattr(nn, self.activation)(m1)
        m1 = nn.Dropout(rate=self.dropout)(m1, deterministic=not train)


        l = self.shared_link.apply(m1, m1, train)

        n0 = jnp.concatenate([m1, l], axis=-1)
        n1 = nn.Conv(features=32, kernel_size=(3, 3))(n0)
        n1 = nn.BatchNorm(use_running_average=not train)(n1)
        n1 = getattr(nn, self.activation)(n1)
        n1 = nn.Dropout(rate=self.dropout)(n1, deterministic=not train)

        # Decoder
        d0 = DecoderBlock(
            features=32, dropout=self.dropout, activation=self.activation
        )(n1, train)
        d0 = jnp.concatenate([d0, e0, f0], axis=-1)
        d1 = DecoderBlock(
            features=32, dropout=self.dropout, activation=self.activation
        )(d0, train)
        
        # Scale x2
        alpha = self.param("alpha", nn.initializers.ones, ())
        x2 = alpha * x2

        o0 = jnp.concatenate([x1, -d1, -x2], axis=-1)
        o1 = nn.Conv(features=1, kernel_size=(1, 1))(o0)

        return o1


class TrainState(train_state.TrainState):
    key: jax.Array
    loss: jnp.ndarray
    batch_stats: Any


if u.DIMS == 2:
    N = u.N[:2]
    DX = u.DX[:2]

domain = Domain(N, DX)
medium = Medium(domain=domain, sound_speed=jnp.ones(N) * u.C, pml_size=u.PML_MARGIN[0])
time_axis = TimeAxis.from_medium(medium, cfl=u.CFL)

sensors_obj = BLISensors(positions=np.load(u.file(u.sensors_path, 0)), n=domain.N)


def simulate(medium, time_axis, p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors_obj)


batch_compiled_simulate = vmap(simulate, in_axes=(None, None, 0))


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
    P_pred, AT = jax.vjp(A, P0_r, c_r)
    residual = P_pred - jnp.expand_dims(P_data, -1)
    d_P0, d_c = AT(residual)
    # d_mu = jnp.expand_dims(d_mu, 0)
    d_c = jnp.expand_dims(d_c, 0)
    return P_pred, d_P0, d_c


# @partial(jit, static_argnums=(2,3))
def step_simple(x, dx, opt_x, opt_x_state):
    x_updates, opt_x_state = opt_x.update(dx, opt_x_state)
    x = optax.apply_updates(x, x_updates)
    return x, opt_x_state

# --------------------------------------------

def train_R(cont=False):
    global exit_flag
    root_key = jax.random.PRNGKey(0)
    main_key, params_key, dropout_key = random.split(root_key, 3)

    # --------------------------------------------

    # @partial(jit, static_argnums=(0))
    def train_step(j, data, state, dropout_key):

        mu = data["mu"]
        ATT_masks = data["ATT_masks"]
        ATT_masks = FourierSeries(jnp.expand_dims(ATT_masks, -1), domain)
        c = data["c"]
        P_data = data["P_data"]
        state_R_mu = state["state_R_mu"]
        state_R_c = state["state_R_c"]
        loss_P_datas = []
        recon = defaultdict(dict)

        mu_r = jnp.zeros((1, *N, 1))
        P0_r = jnp.zeros((u.NUM_LIGHTING_ANGLES, *N, 1))
        c_r = jnp.ones((1, *N, 1)) * u.C

        # opt_mu = optax.adam(learning_rate=u.LR_MU_R)
        # opt_mu_state = opt_mu.init(mu_r)
        # opt_c = optax.adam(learning_rate=u.LR_C_R)
        # opt_c_state = opt_c.init(c_r)

        mu_rs = [mu_r]
        P0_rs = [P0_r]
        c_rs = [c_r]

        dropout_train_key = jax.random.fold_in(
            key=dropout_key, data=state["state_R_mu"].step
        )

        # initial reconstruction
        P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
        loss_P_datas.append(mse(P_pred.squeeze(), P_data))

        d_mu = jnp.mean(d_P0.on_grid / ATT_masks.on_grid, axis=0)
        mu_r = mu_r - u.LR_MU_R * d_mu
        mu_r = jnp.clip(mu_r, 0)
        c_r = c_r - u.LR_C_R * d_c
        # mu_updates, opt_mu_state = opt_mu.update(d_mu, opt_mu_state)
        # mu_r = optax.apply_updates(mu_r, mu_updates)

        # c_updates, opt_c_state = opt_c.update(d_c, opt_c_state)
        # c_r = optax.apply_updates(c_r, c_updates)

        for i in range(1, u.RECON_ITERATIONS):
            P_pred, d_P0, d_c = ATr(P0_r, c_r[0], P_data)
            loss_P_datas.append(mse(P_pred.squeeze(), P_data))

            def loss_R_mu_fn(params, batch_stats):
                P0_r, updates = R_mu.apply(
                    params,
                    (mu_r * ATT_masks).on_grid,
                    d_P0.on_grid,
                    train=True,
                    rngs={"dropout": dropout_train_key},
                    mutable=["batch_stats"],
                )
                batch_stats = updates["batch_stats"]
                mu_r_ = jnp.mean(P0_r / ATT_masks.on_grid, axis=0)
                # mu_updates, opt_mu_state_ = opt_mu.update(d_mu, opt_mu_state)
                # mu_r_ = optax.apply_updates(mu_r, mu_updates)
                return mse(mu_r_, mu), (mu_r_, batch_stats)

            def loss_R_c_fn(params, batch_stats):
                c_r_, updates = R_c.apply(
                    params,
                    c_r,
                    d_c,
                    train=True,
                    rngs={"dropout": dropout_train_key},
                    mutable=["batch_stats"],
                )
                batch_stats = updates["batch_stats"]
                # c_updates, opt_c_state_ = opt_c.update(d_c_, opt_c_state)
                # c_r_ = optax.apply_updates(c_r, c_updates)
                return mse(c_r_, c), (c_r_, batch_stats)

            (loss_R_mu, (mu_r, updates_mu_bs)), d_R_mu = value_and_grad(
                loss_R_mu_fn, argnums=0, has_aux=True
            )(
                state_R_mu.params,
                state["state_R_mu"].batch_stats,
            )

            (loss_R_c, (c_r, updates_c_bs)), d_R_c = value_and_grad(
                loss_R_c_fn, argnums=0, has_aux=True
            )(
                state_R_c.params,
                state["state_R_c"].batch_stats,
            )

            # update reg params
            state_R_mu = state_R_mu.apply_gradients(grads=d_R_mu)
            state_R_c = state_R_c.apply_gradients(grads=d_R_c)

            state_R_mu = state_R_mu.replace(loss=jnp.append(state_R_mu.loss, loss_R_mu))
            state_R_mu = state_R_mu.replace(batch_stats=updates_mu_bs)
            state_R_c = state_R_c.replace(loss=jnp.append(state_R_c.loss, loss_R_c))
            state_R_c = state_R_c.replace(batch_stats=updates_c_bs)

            # reconstruction step
            mu_r = jnp.clip(mu_r, 0)  # Apply non-negativity constraint

            mu_rs.append(mu_r)
            c_rs.append(c_r)

        state["state_R_mu"] = state_R_mu
        state["state_R_c"] = state_R_c
        state["loss_P_datas"][j] = loss_P_datas

        recon["mu_rs"] = mu_rs
        recon["c_rs"] = c_rs

        return state, recon

    dataset = PADataset()
    train_data = [dataset[i] for i in range(u.TRAIN_FILE_START, u.TRAIN_FILE_END)]

    # --------------------------------------------

    def create_train_state(key, model, learning_rate, shape=(1, *N, 1)):
        k1, k2 = random.split(key)
        params = model.init(
            k1,
            random.normal(k2, shape),
            random.normal(k2, shape),
            train=False,
        )
        batch_stats = params["batch_stats"]
        tx = optax.adam(learning_rate)
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            key=dropout_key,
            batch_stats=batch_stats,
            tx=tx,
            loss=0,
        )
    state = defaultdict(dict)
    key1, key2, key3 = random.split(params_key, 3)
    shared_link = SharedLink(dropout=u.DROPOUT, features=32, activation="relu")



    R_mu = DualEncode(dropout=u.DROPOUT, activation="relu", shared_link=shared_link)
    R_c = DualEncode(dropout=u.DROPOUT, activation="relu", shared_link=shared_link)
   
    x1 = jnp.ones((1, 64, 64, 3))
    x2 = jnp.ones((1, 64, 64, 3))
    params = R_mu.init(jax.random.PRNGKey(0), x1, x2, train=True)
    state["shared_link"] = TrainState.create(apply_fn=shared_link.apply, params=params, tx=optax.adam(1e-3))
   
    state["state_R_mu"] = create_train_state(key1, R_mu, u.LR_R_MU, shape=(u.NUM_LIGHTING_ANGLES, *N, 1))
    state["state_R_c"] = create_train_state(key2, R_c, u.LR_R_C)

    # checkpoints
    # restore_args = checkpoint_utils.construct_restore_args(state["state_R_mu"])
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    mngr = ocp.CheckpointManager(
        os.path.abspath(u.checkpoints_path),
        item_names=("state_R_mu", "state_R_c", "loss_P_datas"),
        options=options,
    )
    latest_step = mngr.latest_step()
    if cont and latest_step is not None:
        try:
            restored = mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    state_R_mu=ocp.args.PyTreeRestore(),
                    state_R_c=ocp.args.PyTreeRestore(),
                    state_shared_link=ocp.args.PyTreeRestore(),
                    loss_P_datas=ocp.args.StandardRestore(),
                ),
            )
            print(f"Restored checkpoint {u.checkpoints_path}/{latest_step}")
            state["state_R_mu"].replace(
                params=restored["state_R_mu"]["params"],
                batch_stats=restored["state_R_mu"]["batch_stats"],
                loss=restored["state_R_mu"]["loss"],
                step=restored["state_R_mu"]["step"],
            )
            state["state_R_c"].replace(
                params=restored["state_R_c"]["params"],
                batch_stats=restored["state_R_c"]["batch_stats"],
                loss=restored["state_R_c"]["loss"],
                step=restored["state_R_c"]["step"],
            )
            state["shared_link"].replace(
                params=restored["shared_link"]["params"],
                batch_stats=restored["shared_link"]["batch_stats"],
                loss=restored["shared_link"]["loss"],
                step=restored["shared_link"]["step"],
            )
        except Exception as e:
            print(
                f"Couldn't load checkpoint {u.checkpoints_path}/{latest_step}\n Run again without -c flag to start from scratch"
            )
            print(f"mngr error: {e}")
            exit_flag = True
    else:
        ocp.test_utils.erase_and_create_empty(u.checkpoints_path)
        print("Created empty checkpoint folder")

    # --------------------------------------------

    for data in train_data:
        if exit_flag:
            break

        j = data["file_idx"]

        state, recon = train_step(j, data, state, dropout_key)

        save_recon(j, recon)

        print_losses(j, state)

        latest_step = mngr.latest_step()
        if latest_step is None:
            latest_step = -1
        mngr.save(
            latest_step + 1,
            args=ocp.args.Composite(
                state_R_mu=ocp.args.PyTreeSave(state["state_R_mu"]),
                state_R_c=ocp.args.PyTreeSave(state["state_R_c"]),
                shared_link=ocp.args.PyTreeSave(state["shared_link"]),
                loss_P_datas=ocp.args.StandardSave(state["loss_P_datas"]),
            ),
        )
        mngr.wait_until_finished()


# --------------------------------------------
    def print_nets():
        def print_net(net):
            print(
                net.tabulate(
                    jax.random.key(0),
                    jnp.ones((1, *N, 1)),
                    jnp.ones((1, *N, 1)),
                    compute_flops=True,
                    compute_vjp_flops=True,
                )
            )

        print_net(R_mu)
        # print_net(R_c)

if __name__ == "__main__":
    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="t", nargs="?")
    parser.add_argument("-c", dest="c", action="store_true")
    args = parser.parse_args()
    if args.mode == "t":
        print(f"Continue training: {args.c}")
        train_R(args.c)
        # if args.cont:
        # save params.yaml

    elif args.mode == "r":
        reconstruct_no_reg_opt(0)
    elif args.mode == "p":
        print_nets()
