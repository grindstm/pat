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


def print_recon_losses(j, state):
    print(f"File {j} \nIter\tLoss_P_data\tLoss_mu\t\tLoss_c")
    for i in range(u.RECON_ITERATIONS):
        print(
            f"{i}\t{state[j]['loss_P_datas'][i]:.6f}\t{state[j]['loss_mu'][i]:.4f}\t\t{state[j]['loss_c'][i]:.4f}"
        )


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

    mu_r = jnp.zeros((1, *N, 1))
    c_r = jnp.ones((1, *N, 1)) * u.C

    opt_mu = optax.adam(learning_rate=u.LR_MU_R)
    opt_mu_state = opt_mu.init(mu_r)
    opt_c = optax.adam(learning_rate=u.LR_C_R)
    opt_c_state = opt_c.init(c_r)

    for i in range(u.RECON_ITERATIONS):
        loss, (d_mu, d_c) = ATr_loss(mu_r[0], c_r[0], ATT_masks, P_data)
        mu_r, opt_mu_state = step_simple(mu_r, d_mu, opt_mu, opt_mu_state)
        c_r, opt_c_state = step_simple(c_r, d_c, opt_c, opt_c_state)

        mu_r = jnp.clip(mu_r, 0)  # Apply non-negativity constraint

        state[j]["loss_P_datas"].append(loss)
        state[j]["mu_rs"].append(mu_r)
        state[j]["c_rs"].append(c_r)
        state[j]["loss_mu"].append(mse(mu_r.squeeze(), mu))
        state[j]["loss_c"].append(mse(c_r.squeeze(), c))

    print_recon_losses(j, state)
    save_recon(j, state)


def reconstruct(data):

    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    ATT_masks = FourierSeries(jnp.expand_dims(ATT_masks, -1), domain)
    c = data["c"]
    P_data = data["P_data"]
    state_R_mu = state["state_R_mu"]
    state_R_c = state["state_R_c"]
    loss_P_datas = []

    mu_r = jnp.zeros((1, *N, 1))
    c_r = jnp.ones((1, *N, 1)) * u.C

    opt_mu = optax.adam(learning_rate=u.LR_MU_R)
    opt_mu_state = opt_mu.init(mu_r)
    opt_c = optax.adam(learning_rate=u.LR_C_R)
    opt_c_state = opt_c.init(c_r)

    mu_rs = [mu_r]
    c_rs = [c_r]

    def loss_fn(params, model, x, dx, x_true, opt_x, opt_x_state, batch_stats):
        x, opt_x_state, updates = step(
            params,
            model,
            x,
            dx,
            opt_x,
            opt_x_state,
            train=False,
            rngs={"dropout": dropout_train_key},
            batch_stats=batch_stats,
        )
        return mse(x, x_true), (x, opt_x_state, updates)

    for i in range(u.RECON_ITERATIONS):
        loss, (d_mu, d_c) = ATr_loss(mu_r[0], c_r[0], ATT_masks, P_data)
        loss_P_datas.append(loss)

        mu_loss, (mu_r, opt_mu_state, updates_mu) = loss_fn(
            state_R_mu.params,
            R_mu,
            mu_r,
            d_mu,
            mu,
            opt_mu,
            opt_mu_state,
            state["state_R_mu"].batch_stats,
        )

        # (loss_R_mu, (mu_r, opt_mu_state, updates_mu)), d_R_mu = value_and_grad(
        #     loss_fn, argnums=0, has_aux=True
        # )(
        #     state_R_mu.params,
        #     R_mu,
        #     mu_r,
        #     d_mu,
        #     mu,
        #     opt_mu,
        #     opt_mu_state,
        #     state["state_R_mu"].batch_stats,
        # )

        # (loss_R_c, (c_r, opt_c_state, updates_c)), d_R_c = value_and_grad(
        #     loss_fn, argnums=0, has_aux=True
        # )(
        #     state_R_c.params,
        #     R_c,
        #     c_r,
        #     d_c,
        #     c,
        #     opt_c,
        #     opt_c_state,
        #     state["state_R_c"].batch_stats,
        # )

        # # update reg params
        # state_R_mu = state_R_mu.apply_gradients(grads=d_R_mu)
        # state_R_c = state_R_c.apply_gradients(grads=d_R_c)

        # state_R_mu = state_R_mu.replace(loss=jnp.append(state_R_mu.loss, loss_R_mu))
        # state_R_mu = state_R_mu.replace(batch_stats=updates_mu)
        # state_R_c = state_R_c.replace(loss=jnp.append(state_R_c.loss, loss_R_c))
        # state_R_c = state_R_c.replace(batch_stats=updates_c)

        # reconstruction step
        mu_r = jnp.clip(mu_r, 0)  # Apply non-negativity constraint

        mu_rs.append(mu_r)
        c_rs.append(c_r)

    state["state_R_mu"] = state_R_mu
    state["state_R_c"] = state_R_c
    state[j] = {
        "mu_rs": mu_rs,
        "c_rs": c_rs,
        "loss_P_datas": loss_P_datas,
    }
    return state


def reconstruct_batch(batch):
    pass
    root_key = jax.random.PRNGKey(0)
    main_key, params_key, dropout_key = random.split(root_key, 3)

    def create_train_state(key, model, learning_rate):
        k1, k2 = random.split(key)
        params = model.init(
            k1,
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

    dataset = PADataset()
    train_data = [dataset[i] for i in range(u.TRAIN_FILE_START, u.TRAIN_FILE_END)]

    # --------------------------------------------
    state = defaultdict(dict)
    key1, key2 = random.split(params_key)
    state["state_R_mu"] = create_train_state(key1, R_mu, u.LR_R_MU)
    state["state_R_c"] = create_train_state(key2, R_c, u.LR_R_C)

    # checkpoints
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    mngr = ocp.CheckpointManager(
        os.path.abspath(u.checkpoints_path),
        item_names=("state_R_mu", "state_R_c"),
        options=options,
    )
    latest_step = mngr.latest_step()
    if cont and os.path.exists(u.checkpoints_path + str(latest_step)):
        # ocp.test_utils.erase_and_create_empty(u.checkpoints_path)
        mngr.restore(
            latest_step, args=ocp.args.Composite(state=ocp.args.StandardRestore())
        )
    else:
        ocp.test_utils.erase_and_create_empty(u.checkpoints_path)

    # --------------------------------------------

    for j, data in enumerate(train_data):
        exit_flag = False
        if exit_flag:
            break

        state = train_step(j, data, state, dropout_key)

        save_recon(j, state)

        print_losses(j, state)
