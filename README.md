# Photoacoustic Tomography Image Reconstruction

This repository contains a comprehensive, modular implementation for photoacoustic computed tomography (PACT) image reconstruction. The system supports simultaneous reconstruction of light absorption and sound speed fields using multiple illumination angles and learned regularization in limited view settings.

[**Report:**: "Photoacoustic Tomography Image Reconstruction Simultaneous Reconstruction of Light Absorption and Sound Speed Fields Using Multiple Illumination Angles and Learned Regularization in a Limited View Setting"](report/report.pdf)

[**Presentation**](presentation/presentation.html) 

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: 12GB+ VRAM)
- JAX with CUDA support

### Installation
```bash
# Clone the repository
git clone https://github.com/grindstm/pat.git
cd pat

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Generate synthetic data
python pact_cli.py generate --batch-size 100

# Train regularization networks
python pact_cli.py train --model ynet --illuminations 10 --iterations 50

# Perform reconstruction
python pact_cli.py reconstruct --method learned_regularization --files 0-10

# Visualize results
python pact_cli.py visualize --interactive --file-index 5
```

## 📁 Project Structure

The codebase has been completely refactored into a modular architecture for better maintainability, testing, and extensibility:

```
PACT/
├── src/                          # Main source code (modular architecture)
│   ├── config/                   # Configuration management
│   │   ├── parameters.py         # ConfigManager class and parameter handling
│   │   └── validation.py         # Parameter validation and type checking
│   ├── data/                     # Data generation and management
│   │   ├── generation.py         # Synthetic data generation functions
│   │   ├── dataset.py            # Enhanced PADataset class
│   │   └── preprocessing.py      # Data preprocessing utilities
│   ├── models/                   # Neural networks and training
│   │   ├── networks.py           # All neural network architectures
│   │   ├── losses.py             # Loss functions and regularization
│   │   └── training.py           # Training workflows and state management
│   ├── reconstruction/           # Reconstruction algorithms
│   │   ├── solvers.py            # Reconstruction solver implementations
│   │   └── optimizers.py         # Advanced optimization strategies
│   ├── visualization/            # Plotting and visualization
│   │   ├── plotting.py           # Publication-quality plotting utilities
│   │   └── interactive.py        # Interactive visualization tools
│   └── utils/                    # Utility functions
│       ├── io.py                 # File I/O operations
│       └── jax_utils.py          # JAX-specific utilities and performance tools
├── pact_cli.py                   # Unified command-line interface
├── params.yaml                   # Configuration file
├── generate_data.py              # Legacy data generation script (still used)
├── vis_*.ipynb                   # Visualization notebooks
├── vis.py                        # 3D visualization interface
└── archive_original/             # Archived original files (optional)
    ├── reconstruct.py            # Original monolithic reconstruction script
    ├── PADataset.py              # Original dataset class
    └── util.py                   # Original utility functions
```

## 🛠️ Core Components

- **`ConfigManager`**: Type-safe, validated parameter management
- **`PADataset`**: Dataset class with caching and validation
- **Synthetic data generation**: Vessel networks, illumination patterns, wave simulation
- **Visualization**: `src/visualization/`

### Neural Networks (`src/models/`)
Available architectures:
- **`TreeNet`**: Multi-field network with skip connections (4 inputs)
- **`TreeNet_P0`**: Illumination-aware variant
- **`YNet`**: Y-shaped dual input network
- **`ConcatNet`**: Concatenation-based multi-field network
- **`StepNet`**: Iterative optimization network
- **`RegNet`**: Regression network for scalar outputs

### Reconstruction Methods (`src/reconstruction/`)
- **`GradientDescentSolver`**: Standard gradient-based reconstruction
- **`LearnedRegularizationSolver`**: Neural network regularization
- **`MultiParameterSolver`**: Simultaneous multi-parameter optimization
- **Advanced optimizers**: Adaptive learning rates, early stopping, gradient clipping

## Command-Line Interface (`pact_cli.py`)

### Data Generation
```bash
# Generate synthetic datasets
python pact_cli.py generate [options]

Options:
  --config PATH         Configuration file (default: params.yaml)
  --batch-size N        Number of datasets to generate
```

### Training
```bash
# Train regularization networks
python pact_cli.py train [options]

Options:
  --model {treenet,ynet,concatnet,stepnet,regnet}  Network architecture
  --features N          Number of base features (default: 32)
  --illuminations N     Number of illumination angles (default: 10)
  --iterations N        Training iterations
  --continue           Continue from checkpoint
```

### Reconstruction
```bash
# Perform image reconstruction
python pact_cli.py reconstruct [options]

Options:
  --method {gradient_descent,learned_regularization,multi_parameter}
  --files RANGE         File indices (e.g., '5' or '0-10')
  --illuminations N     Number of illumination angles
  --iterations N        Reconstruction iterations
  --output DIR          Output directory
```

### Visualization
```bash
# Visualize results
python pact_cli.py visualize [options]

Options:
  --interactive         Launch interactive visualization
  --file-index N        File index to visualize
  --save-plots          Save plots to disk
```

## ⚙️ Configuration (`params.yaml`)

### Data Generation Parameters
```yaml
generate_data:
  batch_size: 1000              # Number of datasets to generate
  N: [128, 128, 128]           # Domain size (power of 2 recommended)
  shrink_factor: 3             # Vessel generation quality factor
  dims: 2                      # Spatial dimensions (2 or 3)
  dx: [0.1e-3, 0.1e-3, 0.1e-3] # Spatial discretization (m)
  
  # Physical parameters
  c: 1450                      # Baseline sound speed (m/s)
  c_blood: 1540               # Blood sound speed (m/s)
  c_variation_amplitude: 10    # Background variation amplitude
  c_periodicity: 2            # Perlin noise periodicity
  
  # Simulation parameters
  cfl: 0.3                    # CFL number for time stepping
  pml_margin: [12, 12, 12]    # PML thickness (each side)
  tissue_margin: [20, 20, 20] # Tissue generation margin
  sensor_margin: [16, 16, 16] # Sensor placement margin
  num_sensors: 128            # Number of sensors
  noise_amplitude: 500000     # Sensor noise amplitude
```

### Illumination Parameters
```yaml
lighting:
  lighting_attenuation: true   # Enable attenuation modeling
  num_lighting_angles: 20     # Number of illumination angles
  attenuation: 0.05           # Attenuation coefficient (1/m)
```

### Reconstruction Parameters
```yaml
reconstruct:
  recon_iterations: 10        # Default reconstruction iterations
  lr_mu_r: 1.0               # Learning rate for absorption coefficient
  lr_c_r: 0.5                # Learning rate for sound speed
  recon_file_start: 0        # Start file index for batch reconstruction
  recon_file_end: 1          # End file index for batch reconstruction
```

### Training Parameters
```yaml
train:
  lr_R_mu: 0.0005            # Regularizer learning rate (absorption)
  lr_R_c: 0.000000001        # Regularizer learning rate (sound speed)
  dropout: 0.4               # Dropout rate for networks
  train_file_start: 0        # Start file index for training
  train_file_end: 800        # End file index for training
```

## 🔬 Reconstruction Methods

### 1. Gradient Descent Reconstruction
Standard optimization-based reconstruction:
```bash
python pact_cli.py reconstruct --method gradient_descent --files 0-5 --iterations 50
```

### 2. Learned Regularization
Neural network-based regularization:
```bash
# First train the regularization network
python pact_cli.py train --model ynet --illuminations 10 --iterations 100

# Then use for reconstruction
python pact_cli.py reconstruct --method learned_regularization --files 0-5
```

### 3. Multi-Parameter Optimization
Simultaneous reconstruction of multiple parameters:
```bash
python pact_cli.py reconstruct --method multi_parameter --files 0-5 --iterations 100
```

## 📊 Visualization and Analysis

### Interactive Visualization
```bash
# 2D interactive dashboard
python pact_cli.py visualize --interactive

# 3D volume visualization
python vis.py
```

### Static Plots
```bash
# Generate comparison plots
python pact_cli.py visualize --file-index 5 --save-plots
```

### Jupyter Notebooks
- **`vis.ipynb`**: Interactive dashboard for 2D results
- **`vis_setup.ipynb`**: Data generation visualization
- **`vis_iterations.ipynb`**: Convergence analysis
- **`vis_illum.ipynb`**: Illumination pattern analysis
- **`vis_animated.ipynb`**: Animation creation

## 🧪 Advanced Features

### Custom Loss Functions
The modular design supports easy addition of custom loss functions:
```python
from src.models.losses import create_loss_and_grad_fn

# Create custom loss with regularization
loss_fn = create_loss_and_grad_fn(
    forward_model=my_forward_model,
    loss_type="composite",
    l2_alpha=0.001,
    tv_alpha=0.01
)
```

### Custom Optimizers
Advanced optimization strategies:
```python
from src.reconstruction.optimizers import create_optimizer_from_config

# Create optimizer with custom schedule
optimizer = create_optimizer_from_config({
    "type": "adam",
    "learning_rate": 0.001,
    "schedule": {
        "type": "cosine",
        "decay_steps": 1000
    },
    "gradient_clip": 1.0
})
```

### Batch Processing
Efficient processing of multiple datasets:
```python
from src.reconstruction.solvers import batch_reconstruct

results = batch_reconstruct(
    solver=my_solver,
    dataset=dataset,
    file_indices=range(0, 100),
    num_iterations=50,
    save_results=True
)
```

## 🔧 Development and Debugging

### Common Issues
If developing on this codebase or as a beginner in the JAX and j-Wave ecosystems, please refer to the [Debugging.md](Debugging.md) file for common errors and solutions.

### Testing
# Validate configuration
```shell
python -c "from src.config.parameters import create_default_config; create_default_config()"
```

### Performance Profiling
```bash
# Enable JAX profiling
python pact_cli.py reconstruct --method gradient_descent --files 0 --verbose
```

## 🤝 Contributing

If developing on this codebase or as a beginner in the JAX and j-Wave ecosystems, please refer to the [Debugging.md](Debugging.md) file for common errors and hints/solutions.

### Adding New Features
1. **New networks**: Add to `src/models/networks.py`
2. **New solvers**: Inherit from `ReconstructionSolver`
3. **New loss functions**: Add to `src/models/losses.py`
4. **New optimizers**: Add to `src/reconstruction/optimizers.py`

---

**Note**: This is a completely refactored version. The original files (`reconstruct.py`, `PADataset.py`, `util.py`) have been archived and can be found in the `archive_original/` directory.