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


def recon_step_ATrmu_p_opt2_p(
    j, data, key, lr=[u.LR_MU_R, u.LR_C_R], num_it=u.RECON_ITERATIONS
):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = FourierSeries(data["ATT_masks"], domain)
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
    recon = defaultdict(dict)

    key, key_mu_init, key_c_init = random.split(key, 3)
    mu_r = random.normal(key_mu_init, im_shape)
    c_r = jnp.ones(im_shape) * u.C + 1.0 * random.normal(key_c_init, im_shape)
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
        losses["c"].append(mse(c_rs[-1], c) / jnp.mean(c))
        losses["mu"].append(mse(mu_rs[-1], mu) / jnp.mean(mu))
        losses["sum_mu_c"].append(losses["c"][-1] + losses["mu"][-1])
    recon["mu_rs"] = mu_rs
    recon["c_rs"] = c_rs
    losses[j] = losses

    return losses, recon


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
    mu_r, ATT_masks_ = mulT(P0_r)
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

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.0 * random.normal(key_c_init, im_shape)

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

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.0 * random.normal(key_c_init, im_shape)

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


def recon_step_ATrmu_opt2(j, data, key, lr=[u.LR_MU_R], num_it=u.RECON_ITERATIONS):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
    recon = defaultdict(dict)

    key, key_c_init, key_mu_init = random.split(key, 3)
    mu_r = jnp.ones(im_shape) + 1.0 * random.normal(key_mu_init, im_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.0 * random.normal(key_c_init, im_shape)

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


def recon_step_ATrmu_opt2_norm(
    j, data, key, lr=[u.LR_MU_R, u.LR_C_R], num_it=u.RECON_ITERATIONS
):
    """
    P_pred, d_mu, d_c = ATr(mu_r, c_r[0], P_data)
    Updates done with a single adam optimizer.
    """
    mu = data["mu"]
    ATT_masks = data["ATT_masks"]
    c = data["c"]
    P_data = data["P_data"]

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
    recon = defaultdict(dict)

    key, key_c_init, key_mu_init = random.split(key, 3)
    mu_r = jnp.ones(im_shape) + 1.0 * random.normal(key_mu_init, im_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.0 * random.normal(key_c_init, im_shape)

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
        d_c_n = d_c / c_std

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

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
    recon = defaultdict(dict)

    # mu_r = jnp.zeros(im_shape)
    key, key_c_init, key_mu_init = random.split(key, 3)
    mu_r = jnp.ones(im_shape) + 1.0 * random.normal(key_mu_init, im_shape)
    c_r = jnp.ones(im_shape) * u.C + 1.0 * random.normal(key_c_init, im_shape)

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
        d_c_n = d_c / c_std

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

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    P0_r = jnp.zeros(P0_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.0 * random.normal(key_c_init, im_shape)

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

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    P0_r = jnp.zeros(P0_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.0 * random.normal(key_c_init, im_shape)

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

    losses = {"data": [], "mu": [], "c": [], "sum_mu_c": []}
    recon = defaultdict(dict)

    mu_r = jnp.zeros(im_shape)
    P0_r = jnp.zeros(P0_shape)
    key, key_c_init = random.split(key)
    c_r = jnp.ones(im_shape) * u.C + 1.0 * random.normal(key_c_init, im_shape)

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


def recon_(func, num_illum=np.inf, num_it=u.RECON_ITERATIONS, **kwargs):
    """
    Calls a reconstruction function for each file in the reconstruction dataset. Optionally, it can be limited to a number of illuminations.
    Saves the reconstructions and prints the losses.
    """
    global exit_flag
    key = random.PRNGKey(43)
    dataset = PADataset()

    illum_indices = (
        np.linspace(0, len(dataset), num_illum, endpoint=False).astype(int)
        if num_illum < np.inf
        else range(len(dataset))
    )

    recon_data = [
        dataset[(i, illum_indices)] for i in range(u.RECON_FILE_START, u.RECON_FILE_END)
    ]
    for data in recon_data:
        if exit_flag:
            break

        j = data["file_idx"]

        losses, recon = func(j, data, key, **kwargs)

        # with jax.profiler.trace(u.profile_dir):
        #     losses, recon = func(j, data, key, **kwargs).block_until_ready()

        save_recon(j, recon)

        print_recon_losses(j, losses)

    return losses, recon
