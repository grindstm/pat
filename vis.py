import sys
import os
import jax.numpy as jnp
import vedo
from vedo import Volume, show
import vedo.plotter  # https://vedo.embl.es/docs/vedo/plotter.html#Plotter
# import numpy as np
# from vedo.colors import color_map
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

vedo.settings.default_backend = "vtk"

if __name__ == "__main__":

    plotter = vedo.Plotter(axes=6, bg="white", size=(1600, 800))

    data_path = "data_/"

    if len(sys.argv) == 2:
        IN_NUM = sys.argv[1]
    else:
        IN_NUM = "0"


    show_items = []

    p0_file = f"{data_path}p0/{IN_NUM}.npy"
    if os.path.exists(p0_file):
        p0 = jnp.load(p0_file)
        p0_vol = Volume(p0).isosurface(0.5).alpha(0.05)
        show_items.append(p0_vol)
        print(f"Loaded {p0_file}")
    else:
        print(f"File {p0_file} does not exist")


    p_r_file = f"{data_path}p_r/{IN_NUM}.npy"
    if os.path.exists(p_r_file):
        p_r = jnp.load(p_r_file)
        p_r_vol = Volume(p_r[...,0]).mode(1)

        # threshold 
        # p_r_vol.threshold(below=0.0005, replace=0)

        # color
        vrange = p_r_vol.scalar_range()
        colors = [
        (vrange[0], [0.0,0.0,1.0]),
        (0,       [1.0,1.0,1.0]),
        (vrange[1], [1.0,0.0,0.0])
        ]
        alpha = [1.0, 0.0, 1.]
        # alpha = [1.0,.9,.9,.9, 0.0,.9,.9,.9, 1.]
        p_r_vol.cmap(colors, alpha=alpha)
        p_r_vol.add_scalarbar(title="p_r", c="k") 

        show_items.append(p_r_vol)
        print(f"Loaded {p_r_file}")
    else:
        print(f"File {p_r_file} does not exist")


    plotter.show(*show_items)
