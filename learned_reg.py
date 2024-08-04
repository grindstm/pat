import os
import argparse
import numpy as np
import jax
from jax import random
from jax import vmap, jit, value_and_grad
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from jwave.geometry import Domain, Medium, BLISensors, TimeAxis
from jwave import simulate_wave_propagation
from jaxdf import FourierSeries
import optax
import util as u
from tqdm import tqdm
from PADataset import PADataset


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

def save_params(params, filepath):
    bytes_output = serialization.to_bytes(params)
    with open(filepath, 'wb') as f:
        f.write(bytes_output)


def load_params(filepath, model, mu_example, res_example):
    with open(filepath, 'rb') as f:
        bytes_input = f.read()
    params = model.init(random.PRNGKey(0), mu_example, res_example)['params']
    return serialization.from_bytes(params, bytes_input)
# --------------------------------------------


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
    return P_pred, d_mu, d_ATT_masks, d_c


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


def recon_step(
    mu_r,
    ATT_masks,
    c_r,
    P_data,
    params_R_mu,
    params_R_c,
):
    """
    One iteration of the reconstruction
    """
    mu_r = FourierSeries(mu_r, domain)  # should this be moved?
    c_r = FourierSeries(c_r, domain)

    P_pred, d_mu, d_ATT_masks, d_c = ATr(mu_r, ATT_masks, c_r, P_data)

    d_mu = jnp.expand_dims(d_mu.on_grid, 0)
    d_c = jnp.expand_dims(d_c.on_grid, 0)
    mu_r = jnp.expand_dims(mu_r.on_grid, 0)
    c_r = jnp.expand_dims(c_r.on_grid, 0)

    mu_r = mu_r - (d_mu + R_mu.apply(params_R_mu, mu_r, d_mu))
    c_r = c_r - (d_c + R_c.apply(params_R_c, c_r, d_c))
    return mu_r, c_r


@jit
def R_mu_loss(params_R_mu, mu_r, d_mu, mu):
    mu_r = mu_r - (d_mu + R_mu.apply(params_R_mu, mu_r, d_mu))
    return jnp.mean(jnp.square(mu_r - mu)) / 2.0


@jit
def R_c_loss(params_R_c, c_r, d_c, c):
    c_r = c_r - (d_c + R_c.apply(params_R_c, c_r, d_c))
    return jnp.mean(jnp.square(c_r - c)) / 2.0


def train_R(cont=False):
    """ """
    dataset = PADataset()
    train_data = [dataset[i] for i in range(len(dataset) // 3 * 2)]

    key1, key2 = random.split(random.key(0))
    params_R_mu = R_mu.init(
        key2, random.normal(key1, (1, *N, 1)), random.normal(key1, (1, *N, 1))
    )
    params_R_c = R_c.init(
        key2, random.normal(key1, (1, *N, 1)), random.normal(key1, (1, *N, 1))
    )
    if cont and os.path.exists(u.params_R_mu_path) and os.path.exists(u.params_R_c_path):
        params_R_mu = load_params(u.params_R_mu_path, R_mu, jnp.ones((1, *N, 1)), jnp.ones((1, *N, 1)))
        params_R_c = load_params(u.params_R_c_path, R_c, jnp.ones((1, *N, 1)), jnp.ones((1, *N, 1)))

    # Optimizers
    lr_R_mu = 1e-3
    opt_R_mu = optax.adam(lr_R_mu)
    opt_R_mu_state = opt_R_mu.init(params_R_mu)

    lr_R_c = 1e-3
    opt_R_c = optax.adam(lr_R_c)
    opt_R_c_state = opt_R_c.init(params_R_c)

    for j, data in enumerate(train_data):
        print(f"Training on {j} \nIter\tLoss_mu\t\tLoss_c\t\tLoss_data")

        mu = data["mu"]
        ATT_masks = data["ATT_masks"]
        ATT_masks = FourierSeries(jnp.expand_dims(ATT_masks, -1), domain)
        c = data["c"]
        P_data = data["P_data"]

        # Reconstruction optimizers
        lr_mu_r = 0.3
        opt_mu_r = optax.adam(lr_mu_r)
        opt_mu_r_state = opt_mu_r.init(jnp.zeros_like(mu))

        lr_c_r = 0.3
        opt_c_r = optax.adam(lr_c_r)
        opt_c_r_state = opt_c_r.init(jnp.ones_like(c) * u.C)

        # Initial reconstruction
        key1, key2 = random.split(random.key(0))
        mu_r_0 = jnp.expand_dims(jnp.zeros_like(mu, dtype=jnp.float32), -1)
        c_r_0 = jnp.expand_dims(jnp.ones_like(c) * u.C, -1)
        mu_r, c_r = recon_step(
            mu_r_0, ATT_masks, c_r_0, P_data, params_R_mu, params_R_c
        )

        mu_rs = [mu_r]
        c_rs = [c_r]
        for i in range(u.RECON_ITERATIONS):
            P_pred, d_mu, d_ATT_masks, d_c = ATr(mu_r[0], ATT_masks, c_r[0], P_data)

            # update mu_r and c_r
            mu_r_updates, opt_mu_r_state = opt_mu_r.update(d_mu.squeeze(), opt_mu_r_state)
            mu_r = optax.apply_updates(mu_r, mu_r_updates.reshape(1, *N, 1))
            mu_rs.append(mu_r)

            c_r_updates, opt_c_r_state = opt_c_r.update(d_c.squeeze(), opt_c_r_state)
            c_r = optax.apply_updates(c_r, c_r_updates.reshape(1, *N, 1))
            c_rs.append(c_r)

            d_mu = jnp.reshape(d_mu, (1, *N, 1))
            d_c = jnp.reshape(d_c, (1, *N, 1))

            loss_R_mu, d_R_mu = value_and_grad(R_mu_loss, argnums=0)(
                params_R_mu, mu_r, d_mu, mu
            )
            loss_R_c, d_R_c = value_and_grad(R_c_loss, argnums=0)(
                params_R_c, c_r, d_c, c
            )

            print(
                f"{i}\t{loss_R_mu:.4f}\t\t{loss_R_c:.4f}\t\t{jnp.mean(jnp.square(P_pred.squeeze() - P_data)):.6f}"
            )

            # update reg params
            params_R_mu_updates, opt_R_mu_state = opt_R_mu.update(
                d_R_mu, opt_R_mu_state
            )
            params_R_mu = optax.apply_updates(params_R_mu, params_R_mu_updates)
            params_R_c_updates, opt_R_c_state = opt_R_c.update(d_R_c, opt_R_c_state)
            params_R_c = optax.apply_updates(params_R_c, params_R_c_updates)

            # save mu_r and c_r
            mu_r_file = u.file(u.mu_r_path, j, i)
            jnp.save(mu_r_file, mu_r.squeeze())

            c_r_file = u.file(u.c_r_path, j, i)
            jnp.save(c_r_file, c_r.squeeze())

    # save reg params
    save_params(params_R_mu, u.params_R_mu_path)
    save_params(params_R_c, u.params_R_c_path)
    # bytes_output = serialization.to_bytes(params_R_mu)
    # with open(u.params_R_mu_path, "wb") as f:
    #     f.write(bytes_output)
    # bytes_output = serialization.to_bytes(params_R_c)
    # with open(u.params_R_c_path, "wb") as f:
    #     f.write(bytes_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="t", nargs="?")
    parser.add_argument("cont", type=bool, default=True, nargs="?")
    args = parser.parse_args()
    if args.mode == "t":
        train_R(args.cont)
    elif args.mode == "r":
        reconstruct()
    else:
        try_this()
