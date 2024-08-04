import os
import jax.numpy as jnp
import numpy as np
import vedo
from vedo import Plotter, Volume, Text2D, Points, Slider2D, color_map
import glob
import util as u
import argparse

vedo.settings.default_backend = "vtk"

class VolumeVisualizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_index = 0
        self.plotter = Plotter(axes=6, bg="white", size=(1600, 800))
        self.max_iteration_index = 0
        self.update_max_indices()
        self.iteration_index = self.max_iteration_index
        self.c_vol_original = Volume()
        self.p_r_vol_original = Volume()
        self.last_thresholded_item = None
        self.items = self.load_data()

    def update_max_indices(self):
        self.max_file_index = len(os.listdir(os.path.join(self.data_path, "p0"))) - 1
        self.max_iteration_index = (
            len(glob.glob(os.path.join(self.data_path, f"p_r/{self.file_index}_*.npy")))
            - 1
        )

    def load_data(self):
        self.plotter.clear()
        items = []

        # Load p0 volume
        p0_file = os.path.join(self.data_path, "p0", f"{self.file_index}.npy")
        p0_vol = self.load_volume(p0_file).isosurface(0.5).alpha(0.1).cmap("Greens")
        items.append(p0_vol)

        # Load p_r volume
        p_r_file = os.path.join(
            self.data_path, "p_r", f"{self.file_index}_{self.iteration_index}.npy"
        )
        p_r_vol = self.load_volume(p_r_file)
        vrange = p_r_vol.scalar_range()
        colors = [
            (vrange[0], [0.0, 0.0, 1.0]),
            (0, [1.0, 1.0, 1.0]),
            (vrange[1], [1.0, 0.0, 0.0]),
        ]
        alpha = [1.0, 0.0, 1.0]
        p_r_vol.cmap(colors, alpha=alpha).add_scalarbar("p_r").mode(1).alpha(0.1)
        p_r_vol.name = "p_r"
        print(p_r_vol.name)
        self.p_r_original = p_r_vol.copy()
        items.append(p_r_vol)

        # p_r sliders
        self.plotter.add_slider(
            self.change_iteration_index,
            0,
            self.max_iteration_index,
            value=self.iteration_index,
            title="Iteration Index",
            pos="top-right",
        )
        # self.plotter.add_slider(
        #     lambda widget, event: self.apply_threshold(widget, event, p_r_vol),
        #     0.0,
        #     0.01,
        #     title="p_r Threshold",
        #     value=0.01,
        #     pos=[(.5,.06), (.7,.06)],
        # )

        # Load sound speed volume
        c_file = os.path.join(self.data_path, "c", f"{self.file_index}.npy")
        lower = u.C - u.SOUND_SPEED_VARIATION_AMPLITUDE / 2
        upper = u.C + u.SOUND_SPEED_VARIATION_AMPLITUDE / 2
        c_vol = self.load_volume(c_file)
        
        # vrange = c_vol.scalar_range()
        colors = [
            (lower - 1, [0.0, 1.0, 0.0]),
            (u.C, [1.0, 1.0, 1.0]),
            (upper + 1, [1.0, 0.0, 1.0]),
        ]
        alpha = [0.01, 0.0,0.01]
        c_vol.cmap(colors, alpha=alpha,  vmin=lower - 1, vmax=upper + 1).add_scalarbar("c", pos=(0.775 - 0.1, 0.05))#.isosurface(np.linspace(lower, upper, 5))
        c_vol.name = "c"
        self.c_vol_original = c_vol.copy()
        self.last_thresholded_item = c_vol
        items.append(c_vol)

        self.plotter.add_slider(
            lambda widget, event: self.apply_threshold(widget, event, c_vol),
            lower, 
            upper,
            title="c Threshold",
            value=upper,
            pos=[(.5,.12), (.7,.12)],
        )

        # Load sensor points
        sensors_file = os.path.join(self.data_path, "sensors", f"{self.file_index}.npy")
        sensor_points = self.load_sensors(sensors_file)
        items.append(sensor_points)

        self.plotter.add(items)
        return items

    def load_volume(self, file_path):
        if os.path.exists(file_path):
            data = jnp.load(file_path)
            volume = Volume(data)
            return volume
        else:
            print(f"File {file_path} does not exist")
            return Volume()

    def load_sensors(self, file_path):
        if os.path.exists(file_path):
            sensors = jnp.load(file_path)
            sensor_points = Points(sensors.T)
            self.plotter.add(sensor_points)
            return sensor_points
        else:
            print(f"File {file_path} does not exist")
            return Points()

    def change_file_index(self, widget, event, increment):
        self.file_index = np.clip(self.file_index + increment, 0, self.max_file_index)
        self.update_max_indices()
        self.items = self.load_data()
        self.plotter.render()

    def change_iteration_index(self, widget, event):
        self.iteration_index = int(widget.value)
        widget.value = self.iteration_index
        self.items = self.load_data()
        self.plotter.render()

    def apply_threshold(self, widget, event, item):
        if item.name == "p_r":
            p_r_data = self.p_r_original.tonumpy()
            p_r_data[p_r_data < widget.value] = 0
            item = Volume(p_r_data)
        elif item.name == "c":
            c_data = self.c_vol_original.tonumpy()
            c_data[c_data < widget.value] = u.C
            c_vol = Volume(c_data)
            lower = u.C - u.SOUND_SPEED_VARIATION_AMPLITUDE / 2
            upper = u.C + u.SOUND_SPEED_VARIATION_AMPLITUDE / 2
            colors = [
            (lower - 1, [0.0, 1.0, 0.0]),
            (u.C, [1.0, 1.0, 1.0]),
            (upper + 1, [1.0, 0.0, 1.0]),
            ]
            alpha = [0.01, 0.0,0.01]
            c_vol.cmap(colors, alpha=alpha,  vmin=lower - 1, vmax=upper + 1).add_scalarbar("c", pos=(0.775 - 0.1, 0.05))#.isosurface(np.linspace(lower, upper, 5))
            c_vol.name = "c"
            item = c_vol
            del c_vol, c_data
        item.threshold(below=widget.value, replace=0)#.mode(1)
        self.plotter.remove(self.last_thresholded_item)
        # self.plotter.remove(c_vol)
        self.plotter.add(item)
        self.plotter.render()
        self.last_thresholded_item = item

    def show_hide_c(self, widget, event):
        c_vol = self.items[-1]
        if c_vol.alpha() == 0.0:
            c_vol.alpha(0.1)
            widget.switch()
        else:
            c_vol.alpha(0.0)
            widget.switch()
        self.plotter.render()

    def show(self):
        self.plotter.add_button(
            lambda widget, event: self.change_file_index(widget, event, -1),
            states=("<"),
            pos=(0.02, 0.05),
        )
        self.plotter.add_button(
            lambda widget, event: self.change_file_index(widget, event, 1),
            states=(">"),
            pos=(0.06, 0.05),
        )
        self.plotter.add_button(
            lambda widget, event: self.show_hide_c(widget, event),
            states=("c is visible", "c is hidden"),
            pos=(0.06, 0.1),
        )
        self.plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", type=str, default=None, help="data path", nargs="?"
    )
    args = parser.parse_args()
    if args.data_path is not None:
        u.DATA_PATH = args.data_path

    visualizer = VolumeVisualizer(u.DATA_PATH)
    visualizer.show()
