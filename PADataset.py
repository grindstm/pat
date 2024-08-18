import os
import numpy as np
import jax.numpy as jnp
import util as u
import generate_data as gd
from jax import vmap


if u.DIMS == 2:
    TISSUE_MARGIN = u.TISSUE_MARGIN[:2]


class PADataset:
    def __init__(self, path=u.DATA_PATH):
        self.path = u.DATA_PATH
        self.data = dict()
        self.num_angles = np.load(u.file(u.angles_path, 0)).shape[0]

    def __len__(self):
        return len(os.listdir(u.P_data_noisy_path))

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            data_idx, illum_indices = idx
        else:
            data_idx = idx
            illum_indices = None

        if data_idx in self.data:
            data = self.data[data_idx]
        else:
            data = dict()
            mu = np.array(
                gd.pad_0_wrapper(
                    gd.generate_mu_2d(np.load(u.file(u.mu_path, data_idx))),
                    TISSUE_MARGIN,
                )
            ).astype(jnp.float32)
            angles = np.load(u.file(u.angles_path, data_idx))
            ATT_masks = np.load(u.file(u.ATT_masks_path, data_idx))
            ATT_masks = jnp.expand_dims(
                vmap(gd.pad_1_wrapper, in_axes=(0, None))(ATT_masks, TISSUE_MARGIN), -1
            )

            P_0 = np.load(u.file(u.P_0_path, data_idx))
            c = np.load(u.file(u.c_path, data_idx))
            P_data = np.load(u.file(u.P_data_path, data_idx))
            P_data_noisy = np.load(u.file(u.P_data_noisy_path, data_idx))
            sensors = np.load(u.file(u.sensors_path, data_idx))
            data.update(
                {
                    "file_idx": data_idx,
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
            self.data[data_idx] = data

        # If illumination indices are provided, return only the corresponding illuminations
        if illum_indices is not None:
            data_subset = {
                "file_idx": data["file_idx"],
                "mu": data["mu"],
                "angles": data["angles"][illum_indices],
                "ATT_masks": data["ATT_masks"][illum_indices],
                "P_0": data["P_0"][illum_indices],
                "c": data["c"],
                "P_data": data["P_data"][illum_indices],
                "P_data_noisy": data["P_data_noisy"][illum_indices],
                "sensors": data["sensors"],
            }
            return data_subset
        else:
            return data

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



    # def __getitem__(self, idx):
    #     if idx in self.data:
    #         return self.data[idx]
    #     else:
    #         self.data[idx] = dict()
    #         mu = np.array(gd.pad_0_wrapper(
    #             gd.generate_mu_2d(np.load(u.file(u.mu_path, idx))), TISSUE_MARGIN
    #         )).astype(jnp.float32)
    #         angles = np.load(u.file(u.angles_path, idx))
    #         # ATT_masks = gd.pad_1_wrapper(np.load(u.file(u.ATT_masks_path, idx)), TISSUE_MARGIN)
    #         ATT_masks = np.load(u.file(u.ATT_masks_path, idx))
    #         ATT_masks = jnp.expand_dims(vmap(gd.pad_1_wrapper, in_axes=(0,None))(ATT_masks, TISSUE_MARGIN),-1)

    #         P_0 = np.load(u.file(u.P_0_path, idx))
    #         c = np.load(u.file(u.c_path, idx))
    #         P_data = np.load(u.file(u.P_data_path, idx))
    #         P_data_noisy = np.load(u.file(u.P_data_noisy_path, idx))
    #         sensors = np.load(u.file(u.sensors_path, idx))
    #         self.data[idx].update(
    #             {
    #                 "file_idx": idx,
    #                 "mu": mu,
    #                 "angles": angles,
    #                 "ATT_masks": ATT_masks,
    #                 "P_0": P_0,
    #                 "c": c,
    #                 "P_data": P_data,
    #                 "P_data_noisy": P_data_noisy,
    #                 "sensors": sensors,
    #             }
    #         )
    #         return self.data[idx]