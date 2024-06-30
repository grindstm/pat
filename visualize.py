import sys
import os
import numpy as np
import jax.numpy as jnp
import vedo
from vedo import Volume, show
import vedo.plotter  # https://vedo.embl.es/docs/vedo/plotter.html#Plotter
from vedo.colors import color_map
vedo.settings.default_backend = "vtk"
plotter = vedo.Plotter(axes=6, bg="white", size=(1600, 800))
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

data_path = "data/"


if __name__ == "__main__":


    if len(sys.argv) == 2:
    #     print("To view file # e.g. 4: python generate_data.py 4")
    #     sys.exit(1)
        IN_NUM = sys.argv[1]
    else:
        IN_NUM = "0"
        
    show_items = []
    # Load p0
    p0_file = f"{data_path}p0/{IN_NUM}.npy"
    if os.path.exists(p0_file):
        p0 = jnp.load(p0_file)
        p0_vol = Volume(p0)
        # p0_vol.cmap(["white", "b", "g", "r"]).mode(1)
        show_items.append(p0_vol)
        print(f"Loaded {p0_file}")
    else:
        print(f"File {p0_file} does not exist")

    # Load sensors        
    sensors_file = f"{data_path}sensors/{IN_NUM}.npy"
    if os.path.exists(sensors_file):
        sensors = jnp.load(sensors_file)
        sensor_points = vedo.Points(sensors.T)
        show_items.append(sensor_points)
        print(f"Loaded {sensors_file}")
    else:
        print(f"File {sensors_file} does not exist")

    # Load p_data
    p_data_file = f"{data_path}p_data/{IN_NUM}.npy"
    if os.path.exists(p_data_file):
        p_data = jnp.load(p_data_file)
        # sensor_width = int(sensors[0][-1] - sensors[0][0])
        # sensor_height = int(sensors[1][-1] - sensors[1][0])
        sensor_spacing = sensors[0][1] - sensors[0][0]
        # vmin=jnp.min(p_data)
        # vmax=jnp.max(p_data)
        vmax = np.max(np.abs(p_data))  # This ensures symmetry around zero
        vmin = -vmax
# 
        p_data_vol = Volume(p_data, origin=[sensors[0][0], sensors[1][0], sensors[2][0]],  spacing=[sensor_spacing, sensor_spacing, .5]).cmap("RdBu", vmin=vmin, vmax=vmax).mode(1)

# 
        # def custom_colormap(x):
        #     if x == 0:
        #         return (1, 1, 1, 0)  # Fully transparent for 0 value (white transparent)
        #     else:
        #         return color_map(x, name='jet', vmin=1, vmax=np.max(p_data))  # Use 'jet' colormap from min>0 to max

        # # Apply the colormap
        # p_data_vol.color(custom_colormap).alpha([0, 1])  # Set alpha gradient from 0 to 1

        # p_data_vol.cmap(["b", "white", "r"]).mode(1)
#   
        # colors = ["blue", "white", "red"]  # Colors at the minimum, middle, and maximum
        # nodes = [0.0, 0.5, 1.0]  # Positions of blue, white, and red in the colormap
        # custom_cmap = LinearSegmentedColormap.from_list("customRdBu", list(zip(nodes, colors)))

        # # Apply the colormap
        # p_data_vol = Volume(p_data, origin=[sensors[0][0], sensors[1][0], sensors[2][0]],
        #                     spacing=[sensor_spacing, sensor_spacing, 1]).cmap(custom_cmap, vmin=vmin, vmax=vmax).mode(1)
#   
        show_items.append(p_data_vol)
        p_data_vol.add_scalarbar()
        print(f"Loaded {p_data_file}")
    else:
        print(f"File {p_data_file} does not exist")
    

    # -------------------------------------------

    # Load p_r
    p_r_file = f"{data_path}p_r/{IN_NUM}.npy"
    if os.path.exists(p_r_file):
        p_r = jnp.load(p_r_file)
        p_r_vol = Volume(p_r)
        # p_r_vol.cmap(["white", "b", "g", "r"]).mode(1)
        show_items.append(p_r_vol)
        print(f"Loaded {p_r_file}")
    else:
        print(f"File {p_r_file} does not exist")
        
    
    plotter.show(*show_items, interactive=True)


