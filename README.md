# Photoacoustic Tomography Image Reconstruction
This repository contains the code used to generate and reconstruct images for the paper "Photoacoustic Tomography Image Reconstruction Simultaneous Reconstruction of Light Absorption and Sound Speed Fields Using Multiple Illumination Angles and Learned Regularization in a Limited View Setting" 


If developing on this codebase or as a beginner in the JAX and j-Wave ecosystems, please refer to the [Debugging.md](Debugging.md) file for common errors and hints/solutions.

# Files of interest

Image reconstruction: `reconstruct.py`

Generating data: `generate_data.py`

Dataset class: `PADataset.py`. Prepares generated and reconstructed images when loading. 

Setting parameters: `params.yaml`. These parameters are read by `util.py`, which is imported to all other scripts as `u`, exposing the parameters. `util.py` also defines some helper functions and establishes important environment flags for using JAX in this workflow. 

Files with names containing `vis` involve visualizing results, running experiments and generating figures:
- `vis.ipynb` provides an interactive "dashboard" for viewing generated and reconstructed images.
- `vis_setup` creates figures for the Data Generation section of the paper.
- `vis.py` creates an interface for viewing 3D setups and results.
- ...

# Workflow
## 0. Setup
To set up a virtual environment and install the necessary dependencies for this project, follow these steps:
#### 1. Clone the Repository
```bash
git clone https://github.com/grindstm/pat.git
cd pat
```
#### 2. Create a Virtual Environment
  ```bash
  python3 -m venv venv
  ```
#### 3. Activate the Virtual Environment
  ```bash
  source venv/bin/activate
  ```
#### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
## 1. Set parameters in `params.yaml`
- Files will be generated in `data_path`. Changing this is a simple way to create/return to and seamlessly work with different datasets. 
- `geometry`
  - `dims: 2`, the first 2 numbers of any 3-tuple are used.
    - e.g. `N: [128, 128, 128]` a 3D volume of 128^3 will be generated and summed to a 128^2 image along the first axis.
  - `geometry/dims: 3` generates 3D images
  - `shrink_factor`: when generating low-resolution images, use an integer  to generate volumes that multiple of `N`. They are then shrunk using spline interpolation by the same factor, decently preserving the quality. This works around the limitations of the VSystem vessel generator.
  - `dx`: Simulation domain spatial discretization
  - `c`: baseline sound speed in fat for generating background Perlin noise
  - `c_periodicity`: rate of spatial repetition of Perlin noise
  - `c_variation_amplitude`: range of baseline sound speed around `c`.
  - `c_blood`: sound speed for blood.
  - `cfl`: Courant–Friedrichs–Lewy used in the simulation when defining the time axis from the medium. Since the sound speed is heterogeneous and being reconstructed, this is a baseline value. Adjust this if the simulations become unstable or better temporal resolution is desired.
  - `pml_margin`: the margin added to each side of the domain the accommodate the perfeclty matched layer (PML) which absorbs waves. Small values result in wrapping of waves around the domain and reflections back into the domain.
  - `tissue_margin`: the margin added to each side of the domain when generating images. For example, with `N: [128, 128, 128]` and `tissue_margin: [20, 20, 20]`, the vessels will be generated in a domain `[88, 88, 88]`. This should be larger than `pml_margin`.
  - `sensor_margin`: defines the distance from the edge of the boundary for sensor positions. This should be larger than `pml_margin`
  - `num_sensors`: in 2D simulations, sensors are created in a line, so this can be any number. Higher counts give better data spatial resolution. In 3D, this number should have an integer square root as the sensors are currently created on a square plane.
  - `noise_amplitude`: the amplitude of smoothed, windowed Gaussian noise added to sensor data to improve the realism of the reconstruction task. 
- `lighting`
  - `num_lighting_angles`: the number of illuminations used to generate images. Attenuation masks and simulation data are created for each angle. Angles are evenly distributed in a circle around the domain, with an equal level of absorption at the center of a square domain.
  - `attenuation`: the attenuation coefficient $\mu$ used in Eq. 2. $p_0 = \mu(\mathbf{x}) I_0 e^{-\mu(\mathbf{x}) d(\mathbf{x}, \phi)}$, where $I_0$ is clipped to 1 and the light is projected as if from a line or plane into the domain from the circle or sphere with its edge at the tissue margin.
- `reconstruct`
  - `recon_iterations`: a default value for reconstruction iterations. This value is often overridden as reconstruction functions are called directly.
  - `lr_mu_r`, `lr_c_r`: default values for learning rates used by the `mu_r`, `c_r` optimizers.
  - `recon_file_start`, `recon_file_end`: indices of files to be reconstructed in a batch when running `python reconstruct.py r2`. 
- `train`
  - `lr_R_mu`, `lr_R_c`: default values for learning rates used by the regularizer parameter optimizers.
  - `train_file_start`, `train_file_end`: indices of files to be reconstructed in a batch when running `python reconstruct.py -t`.
## 2. Run `generate_data.py`
Run `python generate_data.py`. Many operations are parallelized using JAX, and some guardrails are in place to prevent GPU memory exhaustion, however a minimum of 12GB of VRAM is recommended. 
## 3. Reconstruct with `reconstruct.py`
Use argument `r1` to reconstruct using 1 optimizer and `r2` to reconstruct using 2. Gradients are shared during these optimizations. The number of illuminations and reconstruction iterations can be set, e.g.: `python reconstruct.py r2 -l=5 -i=10` for 5 illuminations (drawn, evenly spaced from those generated) and 10 iterations. This will load any existing trained parameters and reconstruct the images in the `data_path` based on the `recon_file_start` and `recon_file_end` parameters. Argument `r3` reconstructs using the learned regularizer by loading the latest checkpoint in the checkpoints folder, which is automatically created when training. 

`python reconstruct.py t` will train the parameters of the most recent experiment using the files indexed `train_file_start` - `train_file_end`. The `-c` flag will load parameters and continue the last training. When using this, be sure to update `train_file_start` and `train_file_end`. Use `ctrl+c` in the terminal to signal that training should stop. The current file will finish and then training will stop. This prevents JAX from typing up the GPU. 

`python reconstruct.py p` uses from Flax `model.tabulate` to print the construction of the network.

### Changing the model in use
The models are defined `R=...`. Some commented-out examples exist. Note that the call to `create_train_state` in the training function requires the a list of the shapes of the input images, so currently this must be manually changed if, for instance, switching between networks that take $P_0$ and $\mu$. `print_nets` is also not yet smart enough to know when the model has been changed. 

## Visualize the results
The first couple of cells in `vis.ipynb` will display a dashboard that allows you to scrub through the files, illuminations and reconstructions. The other `vis...` files contain experiments and visualizations that were used in the paper.

For 3D images, ensure vedo is installed and run `python vis.py`. 

# Notes
`reconstruct.py` has not been extensively tested on 3D data and there is a strong likelihood of resource exhaustion when using multiple illuminations. 

Please forgive the significant code duplication. This style was adopted to permit the use of `@jit` on functions, though this isn't yet working. 