import sys
import os
import jax.numpy as jnp
import numpy as np
import vedo
from vedo import Volume, show, Slider2D
import vedo.plotter  # https://vedo.embl.es/docs/vedo/plotter.html#Plotter
import glob


vedo.settings.default_backend = "vtk"
data_path = "data_/"
file_index = 0
max_file_index = len(os.listdir(os.path.join(data_path, "p0"))) - 1

iteration_index = 0

plotter = vedo.Plotter(axes=6, bg="white", size=(1600, 800))


# File index buttons


def file_index_prev_button_callback(widget, event):
    global file_index, items
    max_file_index = len(os.listdir(os.path.join(data_path, "p0"))) - 1
    file_index = np.clip(file_index - 1, 0, max_file_index)
    return load_data(data_path, file_index, iteration_index)


# p_r
# threshold slider
def threshold_slider_callback(widget, event):
    threshold = widget.value
    items[1].threshold(below=threshold, replace=0)
    plotter.render()
    # items[1].alpha(widget.value)
    # update_visualization(file_index_slider.value, plotter, data_path)
    # plotter.clear()
    # plotter.show()

    # Iteration index slider


def iteration_index_slider_callback(widget, event):
    widget.value = int(widget.value)
    iteration_index = int(widget.value)
    items = load_data(data_path, file_index, iteration_index)

    plotter.clear()
    plotter.add(items)
    plotter.render()


def load_data(data_path, file_index, iteration_index, plotter):
    plotter.clear()
    max_file_index = len(os.listdir(os.path.join(data_path, "p0"))) - 1
    file_index = np.clip(file_index, 0, max_file_index)
    file_index_text = vedo.Text2D(
        f"File Index: {file_index} / {max_file_index}", pos=(0.08, 0.05)
    )
    plotter.add(file_index_text)

    p0_file = os.path.join(data_path, "p0", f"{file_index}.npy")
    if os.path.exists(p0_file):
        p0 = jnp.load(p0_file)
        p0_vol = Volume(p0).isosurface(0.5).alpha(0.1)

        plotter.add(p0_vol)
    else:
        p0_vol = Volume()
        print(f"File {p0_file} does not exist")

    # p_r_file = os.path.join(data_path, "p_r", f"{file_index}_{iteration_index}.npy")
    # if os.path.exists(p_r_file):
    #     p_r = jnp.load(p_r_file)
    #     max_file_index = len(os.listdir(os.path.join(data_path, "p0"))) - 1

    #     p_r_vol = (
    #         Volume(p_r[..., 0])
    #         .cmap("jet", alpha=[1.0, 0.0, 1.0])
    #         .add_scalarbar(title="p_r", c="k")
    #         .mode(1)
    #     )

    #     max_iteration_index = len(glob.glob(f"{data_path}{file_index}_*.npy")) - 1

    #     iteration_index_slider = plotter.add_slider(
    #         iteration_index_slider_callback,
    #         0,
    #         max_iteration_index,
    #         value=0,
    #         title="Iteration Index",
    #         pos="top-left",
    #     )

    #     threshold_slider = plotter.add_slider(
    #         threshold_slider_callback,
    #         0,
    #         1,
    #         value=0.15,
    #         title="p_r Threshold",
    #         pos="top-right",
    #     )

    #     plotter.add(p_r_vol)
    # else:
    #     p_r_vol = Volume()
    #     print(f"File {p_r_file} does not exist")

    # sensors_file = os.path.join(data_path, "sensors", f"{file_index}.npy")
    # if os.path.exists(sensors_file):
    #     sensors = jnp.load(sensors_file)
    #     sensor_points = vedo.Points(sensors.T)
    #     plotter.add(sensor_points)
    # else:
    #     sensor_points = vedo.Points()
    #     print(f"File {sensors_file} does not exist")

    # return [p0_vol, p_r_vol, sensor_points], [file_index, iteration_index]


if __name__ == "__main__":

    items = load_data(data_path, file_index, iteration_index, plotter)

    def file_index_change(button, event):
        global file_index, max_file_index
        print(dir(button))
        if button.text == ">":
            print("file_index:", file_index)
            file_index += 1
        elif button.text == "<":
            print("file_index:", file_index)
            file_index -= 1
        file_index = np.clip(file_index, 0, max_file_index)
        return load_data(data_path, file_index, iteration_index, plotter)

    file_index_next_button = plotter.add_button(
        file_index_change, states=(">"), pos=(0.06, 0.05)
    )
    file_index_prev_button = plotter.add_button(
        file_index_change, states=("<"), pos=(0.02, 0.05)
    )

    plotter.show()


# def update_threshold(value, item):
#     assert isinstance(item, Volume)
#     item.threshold(below=value, replace=0)
# .isosurface(.2).alpha(0.2)#jnp.max(p_r)*.6)#
# # color
# vrange = p_r_vol.scalar_range()
# colors = [
# (vrange[0], [0.0,0.0,1.0]),
# (0,       [1.0,1.0,1.0]),
# (vrange[1], [1.0,0.0,0.0])
# ]
# # alpha = [1.0,.9,.9,.9, 0.0,.9,.9,.9, 1.]
# p_r_vol.cmap(colors, alpha=alpha)
# alpha = [1.0, 0.0, 1.]
# return


# l_d_params = [data_path, file_index, iteration_index, plotter]
# # items = load_data(data_path, file_index, iteration_index, plotter)
# items = load_data(*l_d_params)

# from functools import partial

# file_index_next_button = plotter.add_button(
#     load_data(*l_d_params, file_increment=1), states=(">"), pos=(0.06, 0.05)
# )
