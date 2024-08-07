import os
import signal
from collections import defaultdict
import argparse
from functools import partial
import numpy as np
import jax
from jax import random
from jax import vmap, jit, value_and_grad
import jax.numpy as jnp
from flax import linen as nn

# from flax import serialization
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
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


def save_recon(j, state):
    for i in range(u.RECON_ITERATIONS):
        mu_r_file = u.file(u.mu_r_path, j, i)
        jnp.save(mu_r_file, state[j]["mu_rs"][i].squeeze())
        c_r_file = u.file(u.c_r_path, j, i)
        jnp.save(c_r_file, state[j]["c_rs"][i].squeeze())


def print_losses(j, state):
    for i in range(u.RECON_ITERATIONS):
        print(
            f"{state['state_R_mu'].loss[j*u.RECON_ITERATIONS + i]:.4f}\t\t{state['state_R_c'].loss[j*u.RECON_ITERATIONS + i]:.4f}\t\t{state[j]['loss_P_datas'][i]:.6f}"
        )


def print_recon_losses(j, state):
    for i in range(u.RECON_ITERATIONS):
        print(f"Training on {j} \nIter\tLoss_mu\t\tLoss_c\t\tLoss_data")
        print(
            f"{state[j]['loss_mu'][i]:.4f}\t\t{state[j]['loss_c'][i]:.4f}\t\t{state[j]['loss_P_datas'][i]:.6f}"
        )


# --------------------------------------------


class EncoderBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class DecoderBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        batch_size, height, width, channels = x.shape
        x = jax.image.resize(
            x, shape=(batch_size, height * 2, width * 2, channels), method="bilinear"
        )
        x = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        return x


class RegularizerCNN(nn.Module):
    @nn.compact
    def __call__(self, x1, x2):
        # Encoder I
        e1 = EncoderBlock(features=16)(x1)
        e2 = EncoderBlock(features=32)(e1)

        # Encoder II
        f1 = EncoderBlock(features=16)(x2)
        f2 = EncoderBlock(features=32)(f1)

        # Merge features
        merged = jnp.concatenate([e2, f2], axis=-1)

        # Decoder
        d1 = DecoderBlock(features=32)(merged)
        d1 = jnp.concatenate([d1, e1], axis=-1)
        d2 = DecoderBlock(features=16)(d1)

        output = nn.Conv(features=1, kernel_size=(1, 1))(d2)
        return output


class TrainState(train_state.TrainState):
    loss: jnp.ndarray


R_mu = RegularizerCNN()
R_c = RegularizerCNN()
# print(
#     R_mu.tabulate(
#         jax.random.key(0),
#         jnp.ones((1, *N, 1)),
#         jnp.ones((1, *N, 1)),
#         compute_flops=True,
#         compute_vjp_flops=True,
#     )
# )


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


def step(params, model, x, dx):
    return x - (dx + model.apply(params, x, dx))


# --------------------------------------------


def train_R(cont=False):
    # train states
    exit_flag = False
    signal.signal(signal.SIGINT, signal_handler)

    def create_train_state(key, model, learning_rate):
        params = model.init(
            key,
            random.normal(key, (1, *N, 1)),
            random.normal(key, (1, *N, 1)),
        )
        tx = optax.adam(learning_rate)
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, loss=0)

    # --------------------------------------------
    def loss(params, model, x, dx, x_true):
        x = step(params, model, x, dx)
        return jnp.mean(jnp.square(x - x_true)) / 2.0

    @partial(jit, static_argnums=(0))
    def train_step(j, data, state):
        mu = data["mu"]
        ATT_masks = data["ATT_masks"]
        ATT_masks = FourierSeries(jnp.expand_dims(ATT_masks, -1), domain)
        c = data["c"]
        P_data = data["P_data"]
        state_R_mu = state["state_R_mu"]
        state_R_c = state["state_R_c"]

        # Initial reconstruction
        mu_r_0 = jnp.zeros((1, *N, 1))
        c_r_0 = jnp.ones((1, *N, 1)) * u.C

        loss_P_datas = []
        P_pred, d_mu, d_ATT_masks, d_c = ATr(mu_r_0[0], ATT_masks, c_r_0[0], P_data)
        loss_P_datas.append(jnp.mean(jnp.square(P_pred.squeeze() - P_data)) / 2.0)
        mu_r = step(state_R_mu.params, R_mu, mu_r_0, d_mu)
        c_r = step(state_R_c.params, R_c, c_r_0, d_c)

        mu_rs = [mu_r]
        c_rs = [c_r]
        for i in range(u.RECON_ITERATIONS):
            P_pred, d_mu, d_ATT_masks, d_c = ATr(mu_r[0], ATT_masks, c_r[0], P_data)
            loss_P_datas.append(jnp.mean(jnp.square(P_pred.squeeze() - P_data)) / 2.0)

            loss_R_mu, d_R_mu = value_and_grad(loss, argnums=0)(
                state_R_mu.params, R_mu, mu_r, d_mu, mu
            )

            loss_R_c, d_R_c = value_and_grad(loss, argnums=0)(
                state_R_c.params, R_c, c_r, d_c, c
            )

            # update reg params
            state_R_mu = state_R_mu.apply_gradients(grads=d_R_mu)
            state_R_c = state_R_c.apply_gradients(grads=d_R_c)

            state_R_mu = state_R_mu.replace(loss=jnp.append(state_R_mu.loss, loss_R_mu))
            state_R_c = state_R_c.replace(loss=jnp.append(state_R_c.loss, loss_R_c))

            # reconstruction step
            mu_r = step(state_R_mu.params, R_mu, mu_r, d_mu)
            mu_rs.append(mu_r)

            c_r = step(state_R_c.params, R_c, c_r, d_c)
            c_rs.append(c_r)

        state["state_R_mu"] = state_R_mu
        state["state_R_c"] = state_R_c
        state[j] = {
            "mu_rs": mu_rs,
            "c_rs": c_rs,
            "loss_P_datas": loss_P_datas,
        }
        return state

    dataset = PADataset()
    train_data = [dataset[i] for i in range(len(dataset) // 3 * 2)]

    # --------------------------------------------

    # checkpoints
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    with ocp.CheckpointManager(
        os.path.abspath(os.path.join(u.DATA_PATH, "checkpoints")),
        options=options,
    ) as mngr:
        latest_step = mngr.latest_step()
        if latest_step is not None and cont:
            state_R_mu = mngr.restore(latest_step, items=state["state_R_mu"])
            state_R_c = mngr.restore(latest_step, items=state["state_R_c"])
        else:
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            state_R_mu = create_train_state(key1, R_mu, u.LR_R_MU)
            state_R_c = create_train_state(key2, R_c, u.LR_R_C)

    # --------------------------------------------

    state = defaultdict(dict)
    state["state_R_mu"] = state_R_mu
    state["state_R_c"] = state_R_c

    for j, data in enumerate(train_data):
        if exit_flag:
            break

        state = train_step(j, data, state)

        save_recon(j, state)

        print_losses(j, state)

        with ocp.CheckpointManager(
            os.path.abspath(os.path.join(u.DATA_PATH, "checkpoints")),
            options=options,
        ) as mngr:
            mngr.save(j, args=ocp.args.StandardSave(state["state_R_mu"]))
            mngr.save(j, args=ocp.args.StandardSave(state["state_R_c"]))
            mngr.wait_until_finished()


# --------------------------------------------


def reconstruct_no_reg(j=0):
    def mse(x, x_true):
        return jnp.mean(jnp.square(x - x_true)) / 2.0

    state = defaultdict(dict)
    state[j] = {
        "loss_mu": [],
        "loss_c": [],
        "loss_P_datas": [],
        "mu_rs": [],
        "c_rs": [],
    }

    dataset = PADataset()
    data = dataset[j]
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    ATT_masks = FourierSeries(jnp.expand_dims(ATT_masks, -1), domain)
    c = data["c"]
    P_data = data["P_data"]

    # Initial reconstruction
    mu_r = jnp.zeros((1, *N, 1))
    c_r = jnp.ones((1, *N, 1)) * u.C

    P_pred, d_mu, d_ATT_masks, d_c = ATr(mu_r[0], ATT_masks, c_r[0], P_data)

    state[j]["loss_P_datas"] = [mse(P_pred.squeeze(), P_data)]

    def step(x, dx, lr):
        return x - lr * dx

    for i in range(u.RECON_ITERATIONS):
        P_pred, d_mu, d_ATT_masks, d_c = ATr(mu_r[0], ATT_masks, c_r[0], P_data)
        state[j]["loss_P_datas"].append(
            jnp.mean(jnp.square(P_pred.squeeze() - P_data)) / 2.0
        )
        mu_r = step(mu_r, d_mu, u.LR_MU_R)
        c_r = step(c_r, d_c, u.LR_C_R)

        state[j]["mu_rs"].append(mu_r)
        state[j]["c_rs"].append(c_r)
        state[j]["loss_mu"].append(mse(mu_r.squeeze(), mu))
        state[j]["loss_c"].append(mse(c_r.squeeze(), c))

    print_recon_losses(j, state)
    save_recon(j, state)


def reconstruct_no_reg_opt(j=0):
    dataset = PADataset()
    data = dataset[j]
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    ATT_masks = FourierSeries(jnp.expand_dims(ATT_masks, -1), domain)
    c = data["c"]
    P_data = data["P_data"]

    state = defaultdict(dict)
    state[j] = {
        "loss_mu": [],
        "loss_c": [],
        "loss_P_datas": [],
        "mu_rs": [],
        "c_rs": [],
    }

    def mse(x, x_true):
        return jnp.mean(jnp.square(x - x_true)) / 2.0

    @jit
    def ATr_loss(mu, c, ATT_masks, P_data):
        P_pred, d_mu, d_ATT_masks, d_c = ATr(mu, ATT_masks, c, P_data)
        residual = P_pred - jnp.expand_dims(P_data, -1)
        loss = mse(P_pred.squeeze(), P_data)
        return loss, (d_mu, d_c)

    mu_r = jnp.zeros((1, *N, 1))
    c_r = jnp.ones((1, *N, 1)) * u.C

    optimizer = optax.adam(learning_rate=0.5)
    opt_state = optimizer.init((mu_r, c_r))

    for i in range(u.RECON_ITERATIONS):
        loss, (grad_mu, grad_c) = ATr_loss(mu_r[0], c_r[0], ATT_masks, P_data)

        grad_mu = grad_mu.reshape((1, *N, 1))
        grad_c = grad_c.reshape((1, *N, 1))
        updates, opt_state = optimizer.update((grad_mu, grad_c), opt_state)
        mu_r, c_r = optax.apply_updates((mu_r, c_r), updates)
        mu_r = jnp.clip(mu_r, 0)  # Apply non-negativity constraint

        state[j]["loss_P_datas"].append(loss)
        state[j]["mu_rs"].append(mu_r)
        state[j]["c_rs"].append(c_r)
        state[j]["loss_mu"].append(mse(mu_r.squeeze(), mu))
        state[j]["loss_c"].append(mse(c_r.squeeze(), c))

    print_recon_losses(j, state)
    save_recon(j, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="x", nargs="?")
    parser.add_argument("cont", type=bool, default=False, nargs="?")
    args = parser.parse_args()
    if args.mode == "t":
        train_R(args.cont)
    elif args.mode == "r":
        reconstruct()
    else:
        reconstruct_no_reg_opt(0)
