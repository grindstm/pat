import os
import jax.numpy as jnp
import numpy as np
import vedo
from vedo import Plotter, Volume, Text2D, Points, Slider2D, color_map
import glob
import util

vedo.settings.default_backend = "vtk"
BATCH_SIZE, N, DX, C, CFL, PML_MARGIN, TISSUE_MARGIN, SENSOR_MARGIN, NUM_SENSORS, SOUND_SPEED_PERIODICITY, SOUND_SPEED_VARIATION_AMPLITUDE = util.parse_params()
class VolumeVisualizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_index = 0
        self.plotter = Plotter(axes=6, bg='white', size=(1600, 800))
        self.max_iteration_index = 0
        self.update_max_indices()
        self.iteration_index = self.max_iteration_index
        self.items = self.load_data()

    def update_max_indices(self):
        self.max_file_index = len(os.listdir(os.path.join(self.data_path, "p0"))) - 1
        self.max_iteration_index = len(glob.glob(os.path.join(self.data_path, f"p_r/{self.file_index}_*.npy"))) - 1

    def load_data(self):
        self.plotter.clear()
        items = []

        # Load p0 volume
        p0_file = os.path.join(self.data_path, "p0", f"{self.file_index}.npy")
        p0_vol = self.load_volume(p0_file).isosurface(0.5).alpha(0.1).cmap("Greens")
        items.append(p0_vol)

        # Load p_r volume
        p_r_file = os.path.join(self.data_path, "p_r", f"{self.file_index}_{self.iteration_index}.npy")
        p_r_vol = self.load_volume(p_r_file).cmap("coolwarm").add_scalarbar("p_r").mode(1).alpha(0.1)
        items.append(p_r_vol)

        # # Load sound speed volume
        # c_file = os.path.join(self.data_path, "c", f"{self.file_index}.npy")
        # lower= C-SOUND_SPEED_VARIATION_AMPLITUDE/2
        # upper= C+SOUND_SPEED_VARIATION_AMPLITUDE/2
        # sound_speed_vol = self.load_volume(c_file)
        # sound_speed_vol.alpha(0.01).add_scalarbar("c", pos=(.775-.1,.05)).isosurface(np.linspace(lower,upper,5))
        # items.append(sound_speed_vol)

        # # Load sensor points
        # sensors_file = os.path.join(self.data_path, "sensors", f"{self.file_index}.npy")
        # sensor_points = self.load_sensors(sensors_file)
        # items.append(sensor_points)

        # Add sliders
        self.plotter.add_slider(self.change_iteration_index, 0, self.max_iteration_index, title="Iteration Index", pos='top-right')
        self.plotter.add_slider(lambda widget, event:self.apply_threshold(widget, event, p_r_vol), 0, 1, title="p_r Threshold", value=0.15, pos='right')

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
        self.items = self.load_data()
        self.plotter.render()

    def apply_threshold(self, widget, event, item):
        threshold = widget.value
        # for item in self.items:
            # if isinstance(item, Volume):
        item.threshold(above=threshold, replace=0).mode(1)
        self.plotter.render()

    def show(self):
        self.plotter.add_button(lambda widget, event: self.change_file_index(widget, event, -1), states=("<"), pos=(0.02, 0.05))
        self.plotter.add_button(lambda widget, event: self.change_file_index(widget, event, 1), states=(">"), pos=(0.06, 0.05))
        self.plotter.show()

if __name__ == "__main__":
    visualizer = VolumeVisualizer("data/")
    visualizer.show()
