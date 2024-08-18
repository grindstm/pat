# Files of interest
Image reconstruction: `reconstruct.py`
Generating data: `generate_data.py`
Dataset class: `PADataset.py`. Properly prepares generated and reconstructed images when loading.  
Setting parameters: `params.yaml`. These parameters are read by `util.py`, which is imported to all other scripts as `u`, exposing the parameters. `util.py` also defines some helper functions and establishes important environment flags for using JAX in this workflow. 

Files with names containing `vis` involve visualizing results and generating figures:
- `vis.ipynb` provides an interactive "dashboard" for viewing generated and reconstructed images.
- `vis_setup` creates figures for the Data Generation section of the paper.
- `vis.py` creates an interface for viewing 3D setups and results.
- ...
