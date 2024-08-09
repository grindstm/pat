import os
import numpy as np
import jax.numpy as jnp
import util as u
import generate_data as gd
from jax import vmap


if u.DIMS == 2:
    TISSUE_MARGIN = u.TISSUE_MARGIN[:2]

class PADataset():
    def __init__(self):
        self.data = dict()

    def __len__(self):
        return len(os.listdir(u.P_data_noisy_path))

    def __getitem__(self, idx):
        if idx in self.data:
            return self.data[idx]
        else:
            self.data[idx] = dict()
            mu = np.array(gd.pad_0_wrapper(
                gd.generate_mu_2d(np.load(u.file(u.mu_path, idx))), TISSUE_MARGIN
            )).astype(jnp.float32)
            angles = np.load(u.file(u.angles_path, idx))
            # ATT_masks = gd.pad_1_wrapper(np.load(u.file(u.ATT_masks_path, idx)), TISSUE_MARGIN)
            ATT_masks = np.load(u.file(u.ATT_masks_path, idx))
            ATT_masks = vmap(gd.pad_1_wrapper, in_axes=(0,None))(ATT_masks, TISSUE_MARGIN)

            P_0 = np.load(u.file(u.P_0_path, idx))
            c = np.load(u.file(u.c_path, idx))
            P_data = np.load(u.file(u.P_data_path, idx))
            P_data_noisy = np.load(u.file(u.P_data_noisy_path, idx))
            sensors = np.load(u.file(u.sensors_path, idx))
            self.data[idx].update(
                {
                    "file_idx": idx,
                    "mu": mu,
                    "angles": angles,
                    "ATT_masks": ATT_masks,
                    "P_0": P_0,
                    "c": c,
                    "P_data": P_data,
                    "P_data_noisy": P_data_noisy,
                    "sensors": sensors,
                }
            )
            return self.data[idx]

    def load_recon(self, idx, it):
        if os.path.exists(u.file(u.mu_r_path, idx, it)):
            mu_r = np.load(u.file(u.mu_r_path, idx, it))
        else:
            mu_r = np.zeros_like(self.data[idx]["mu"])
        if os.path.exists(u.file(u.c_r_path, idx, it)):
            c_r = np.load(u.file(u.c_r_path, idx, it))
        else:
            c_r = np.zeros_like(self.data[idx]["c"])
        self.data[idx].update({"mu_r": mu_r, "c_r": c_r})
        return self.data[idx]


