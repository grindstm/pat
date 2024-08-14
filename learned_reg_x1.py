import os
import signal
from collections import defaultdict
import argparse
from typing import Any
from functools import partial
import numpy as np
import jax
from jax import random, lax
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


def print_losses(j, state):
    print(
        "***WARNING***: Training files out of order results in incorrect loss values. Start from file 0 and don't skip. (TODO)"
    )
    print(f"File {j}\nIter\tLoss_P_data\tLoss\tLoss_mu\tLoss_c")

    for i in range(u.RECON_ITERATIONS):
        print(
            f"{i}\t{state['loss_P_datas'][j][i]:.6f}\t{state['state_R'].loss[j*u.RECON_ITERATIONS + i]:.6f}"
            # f"{i}\t{state['loss_P_datas'][j][i]:.6f}\t{state['state_R'].loss[j*u.RECON_ITERATIONS + i]:.6f}\t{state['loss_mu'][j*u.RECON_ITERATIONS + i]:.6f},\t{state['loss_c'][j*u.RECON_ITERATIONS + i]:.6f}"
        )


def print_recon_losses(j, state):
    print(f"File {j} \nIter\tLoss_P_data\tLoss_mu\t\tLoss_c")
    for i in range(u.RECON_ITERATIONS):
        print(
            f"{i}\t{state[j]['loss_P_datas'][i]:.6f}\t{state[j]['loss_mu'][i]:.4f}\t\t{state[j]['loss_c'][i]:.4f}"
        )


# --------------------------------------------


class EncoderBlock(nn.Module):
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return x

        # x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))


class EncodeDense(nn.Module):
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(features=self.features)(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        # x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        return x


class DecoderBlock(nn.Module):
    features: int
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.ConvTranspose(
            features=self.features, kernel_size=(3, 3), padding="SAME"
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        x = nn.ConvTranspose(
            features=self.features, kernel_size=(3, 3), strides=(2, 2)
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return x


class ReconNet_1(nn.Module):
    dropout: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, mu_r, d_mu, c_r, d_c, train: bool = True):
        # concatenate
        combined = jnp.concatenate([mu_r, d_mu, c_r, d_c], axis=-1)

        # Encode
        e0 = EncoderBlock(
            features=32, dropout=self.dropout, activation=self.activation
        )(combined, train)
        e1 = EncoderBlock(
            features=64, dropout=self.dropout, activation=self.activation
        )(e0, train)
        e2 = EncoderBlock(
            features=128, dropout=self.dropout, activation=self.activation
        )(e1, train)
        e3 = EncoderBlock(
            features=256, dropout=self.dropout, activation=self.activation
        )(e2, train)
        e4 = EncoderBlock(
            features=512, dropout=self.dropout, activation=self.activation
        )(e3, train)

        # Flatten
        # flat = jnp.reshape(e4, (e4.shape[0], -1))
        # flat_length = flat.shape[1]

        # Dense
        # dense_0 = EncodeDense(
        #     features=flat_length, dropout=self.dropout, activation=self.activation
        # )(flat, train)
        # dense_1 = EncodeDense(
        #     features=flat_length // 2, dropout=self.dropout, activation=self.activation
        # )(dense_0, train)
        # dense_2 = EncodeDense(
        #     features=flat_length, dropout=self.dropout, activation=self.activation
        # )(dense_1, train)

        # reshape for decoder
        # dense_2 = jnp.reshape(dense_2, (e4.shape[0], *e4.shape[1:]))

        # Decoder
        d0 = DecoderBlock(
            features=256, dropout=self.dropout, activation=self.activation
        )(e4, train)
        d0 = jnp.concatenate([d0, e3], axis=-1)
        d1 = DecoderBlock(
            features=128, dropout=self.dropout, activation=self.activation
        )(d0, train)
        d1 = jnp.concatenate([d1, e2], axis=-1)
        d2 = DecoderBlock(
            features=64, dropout=self.dropout, activation=self.activation
        )(d1, train)
        d2 = jnp.concatenate([d2, e1], axis=-1)
        d3 = DecoderBlock(
            features=32, dropout=self.dropout, activation=self.activation
        )(d2, train)
        d3 = jnp.concatenate([d3, e0], axis=-1)
        d4 = DecoderBlock(
            features=32, dropout=self.dropout, activation=self.activation
        )(d3, train)

        output_images = jnp.split(d4, indices_or_sections=4, axis=-1)
        mu_r_out = nn.Conv(features=1, kernel_size=(1, 1))(output_images[0])
        d_mu_out = nn.Conv(features=1, kernel_size=(1, 1))(output_images[1])
        c_r_out = nn.Conv(features=1, kernel_size=(1, 1))(output_images[2])
        d_c_out = nn.Conv(features=1, kernel_size=(1, 1))(output_images[3])

        alpha_mu = self.param("alpha_mu", nn.initializers.ones, ())
        alpha_mu_broadcasted = alpha_mu * jnp.ones_like(d_mu)
        d_mu_out = alpha_mu_broadcasted * d_mu_out

        alpha_c = self.param("alpha_c", nn.initializers.ones, ())
        alpha_c_broadcasted = alpha_c * jnp.ones_like(d_c)
        d_c_out = alpha_c_broadcasted * d_c_out

        return (mu_r_out - d_mu_out, c_r_out - d_c_out)


class TrainState(train_state.TrainState):
    key: jax.Array
    loss: jnp.ndarray
    batch_stats: Any


R = ReconNet_1(dropout=u.DROPOUT, activation="relu")


def print_nets():
    def print_net(net):
        print(
            net.tabulate(
                jax.random.key(0),
                jnp.ones((1, *N, 1)),
                jnp.ones((1, *N, 1)),
                jnp.ones((1, *N, 1)),
                jnp.ones((1, *N, 1)),
                compute_flops=True,
                compute_vjp_flops=True,
            )
        )

    print_net(R)
    # print_net(R_c)


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
def A(mu, ATT_masks, c):
    P_0 = mu * ATT_masks
    medium = Medium(domain=domain, sound_speed=c, pml_size=u.PML_MARGIN[0])
    return batch_compiled_simulate(medium, time_axis, P_0)


@jit
def ATr(
    mu_r,
    ATT_masks,
    c_r,
    P_data,
):
    P_pred, AT = jax.vjp(A, mu_r, ATT_masks, c_r)
    residual = P_pred - jnp.expand_dims(P_data, -1)
    d_mu, d_ATT_masks, d_c = AT(residual)
    d_mu = jnp.expand_dims(d_mu, 0)
    d_c = jnp.expand_dims(d_c, 0)
    return P_pred, d_mu, d_ATT_masks, d_c


@jit
def ATr_loss(mu, c, ATT_masks, P_data):
    P_pred, d_mu, d_ATT_masks, d_c = ATr(mu, ATT_masks, c, P_data)
    loss = mse(P_pred.squeeze(), P_data)
    return loss, (d_mu, d_c)


# @partial(jit, static_argnums=(2,3))
def step_simple(x, dx, opt_x, opt_x_state):
    x_updates, opt_x_state = opt_x.update(dx, opt_x_state)
    x = optax.apply_updates(x, x_updates)
    return x, opt_x_state


# @partial(jit, static_argnums=(1))
def step(params, model, x, dx, opt_x, opt_x_state, train, rngs, batch_stats):
    dx, updates = model.apply(
        params,  # ['params'], "batch_stats": batch_stats},
        x,
        dx,
        train=train,
        rngs=rngs,
        mutable=["batch_stats"],
    )
    batch_stats = updates["batch_stats"]
    x_updates, opt_x_state = opt_x.update(dx, opt_x_state)
    x = optax.apply_updates(x, x_updates)
    return x, opt_x_state, batch_stats


@jit
def normalize(x):
    min_val = jnp.min(x)
    max_val = jnp.max(x)

    def normalize_case():
        return (x - min_val) / (max_val - min_val), min_val, max_val

    def return_original():
        return x, min_val, max_val

    normalized_x, min_val, max_val = lax.cond(
        jnp.abs(min_val - max_val) < 1e-6, return_original, normalize_case
    )

    return normalized_x, min_val, max_val


@jit
def denormalize(x, min, max):
    return x * (max - min) + min


# --------------------------------------------


def train_R(cont=False):
    global exit_flag
    root_key = jax.random.PRNGKey(0)
    main_key, params_key, dropout_key = random.split(root_key, 3)

    def create_train_state(key, model, learning_rate):
        k1, k2 = random.split(key)
        params = model.init(
            k1,
            random.normal(k2, (1, *N, 1)),
            random.normal(k2, (1, *N, 1)),
            random.normal(k2, (1, *N, 1)),
            random.normal(k2, (1, *N, 1)),
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

    # --------------------------------------------

    # @partial(jit, static_argnums=(0))
    def train_step(j, data, state, dropout_key):

        mu = data["mu"]
        ATT_masks = data["ATT_masks"]
        ATT_masks = FourierSeries(jnp.expand_dims(ATT_masks, -1), domain)
        c = data["c"]
        P_data = data["P_data"]
        state_R = state["state_R"]
        loss_P_datas = []
        recon = defaultdict(dict)

        mu_r = jnp.zeros((1, *N, 1))
        c_r = 1.0 * random.normal(random.PRNGKey(main_key[0]), (1, *N, 1)) + u.C
        # c_r = jnp.ones((1, *N, 1)) * u.C

        mu_rs = [mu_r]
        c_rs = [c_r]

        dropout_train_key = jax.random.fold_in(
            key=dropout_key, data=state["state_R"].step
        )

        # initial reconstruction
        loss, (d_mu, d_c) = ATr_loss(mu_r[0], c_r[0], ATT_masks, P_data)
        loss_P_datas.append(loss)
        mu_r = mu_r - u.LR_MU_R * d_mu
        c_r = c_r - u.LR_C_R * d_c
        mu_r = jnp.clip(mu_r, 0)  # Apply non-negativity constraint
        mu_rs.append(mu_r)
        c_rs.append(c_r)

        for i in range(1, u.RECON_ITERATIONS):
            loss, (d_mu, d_c) = ATr_loss(mu_r[0], c_r[0], ATT_masks, P_data)
            loss_P_datas.append(loss)

            def loss_fn(params, mu_r, d_mu, c_r, d_c):

                # normalized inputs
                n_mu_r, mu_min, mu_max = normalize(mu_r)
                n_c_r, c_min, c_max = normalize(c_r)
                n_d_mu, d_mu_min, d_mu_max = normalize(d_mu)
                n_d_c, d_c_min, d_c_max = normalize(d_c)

                (mu_r_n, c_r_n), updates = R.apply(
                    params,
                    n_mu_r,
                    n_d_mu,
                    n_c_r,
                    n_d_c,
                    train=True,
                    rngs={"dropout": dropout_train_key},
                    mutable=["batch_stats"],
                )
                mu_r = denormalize(mu_r_n, mu_min, mu_max)
                c_r = denormalize(c_r_n, c_min, c_max)

                loss_mu = mse(mu_r, mu)
                loss_c = mse(c_r, c)
                balance = 1.0
                loss = loss_mu + balance * loss_c
                return loss, ((loss_mu, loss_c), mu_r, c_r, updates["batch_stats"])

            (loss, ((loss_mu, loss_c), mu_r, c_r, updates)), d_R = value_and_grad(
                loss_fn, argnums=0, has_aux=True
            )(state["state_R"].params, mu_r, d_mu, c_r, d_c)

            # update reg params
            state_R = state_R.apply_gradients(grads=d_R)

            state_R = state_R.replace(loss=jnp.append(state_R.loss, loss_mu))
            state_R = state_R.replace(batch_stats=updates)

            # reconstruction step
            mu_r = jnp.clip(mu_r, 0)  # Apply non-negativity constraint
            c_r = jnp.clip(c_r, min=1425, max=1575)

            mu_rs.append(mu_r)
            c_rs.append(c_r)

        state["state_R"] = state_R
        # state["state_R_c"] = state_R_c
        state["loss_P_datas"][j] = loss_P_datas
        state["loss_mu"][j] = loss_mu
        state["loss_c"][j] = loss_c

        recon["mu_rs"] = mu_rs
        recon["c_rs"] = c_rs

        return state, recon

    dataset = PADataset()
    train_data = [dataset[i] for i in range(u.TRAIN_FILE_START, u.TRAIN_FILE_END)]

    # --------------------------------------------
    state = defaultdict(dict)
    key1, key2 = random.split(params_key)
    state["state_R"] = create_train_state(key1, R, u.LR_R_MU)

    # checkpoints
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    mngr = ocp.CheckpointManager(
        os.path.abspath(u.checkpoints_path),
        item_names=("state_R", "loss_P_datas", "loss_mu", "loss_c"),
        options=options,
    )
    latest_step = mngr.latest_step()
    if cont and latest_step is not None:
        try:
            restored = mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    state_R=ocp.args.PyTreeRestore(),
                    loss_P_datas=ocp.args.StandardRestore(),
                    loss_mu=ocp.args.StandardRestore(),
                    loss_c=ocp.args.StandardRestore(),
                ),
            )
            print(f"Restored checkpoint {u.checkpoints_path}/{latest_step}")
            state["state_R"].replace(
                params=restored["state_R"]["params"],
                batch_stats=restored["state_R"]["batch_stats"],
                loss=restored["state_R"]["loss"],
                step=restored["state_R"]["step"],
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
                state_R=ocp.args.PyTreeSave(state["state_R"]),
                loss_P_datas=ocp.args.StandardSave(state["loss_P_datas"]),
                loss_mu=ocp.args.StandardSave(state["loss_mu"]),
                loss_c=ocp.args.StandardSave(state["loss_c"]),
            ),
        )
        mngr.wait_until_finished()


# --------------------------------------------


if __name__ == "__main__":
    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="", nargs="?")
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
