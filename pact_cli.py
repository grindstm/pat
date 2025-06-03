"""
Command-line interface for PACT reconstruction system.

This script provides a unified interface for data generation, training,
reconstruction, and visualization tasks in the PACT pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.parameters import create_default_config
from src.data.dataset import PADataset
from src.models.networks import create_network
from src.reconstruction.solvers import create_solver, batch_reconstruct
from src.visualization.plotting import setup_publication_style


def setup_data_generation(args):
    """Set up and run data generation."""
    print("Setting up data generation...")
    
    config = create_default_config(args.config if hasattr(args, 'config') else None)
    
    # Import and run data generation
    import generate_data
    print(f"Generating {config.batch_size} datasets in {config.data_path}")


def setup_training(args):
    """Set up and run training."""
    print("Setting up training...")
    
    config = create_default_config(args.config if hasattr(args, 'config') else None)
    
    # Create dataset
    dataset = PADataset(config)
    
    # Create network
    network = create_network(
        args.model, 
        features=args.features,
        dropout=config.dropout
    )
    
    # Import and set up training
    from src.models.training import TrainingManager
    
    training_manager = TrainingManager(config, network, dataset)
    
    # Run training
    results = training_manager.train_regularizer(
        num_illuminations=args.illuminations,
        learning_rates=[config.lr_r_mu, config.lr_r_c],
        num_iterations=args.iterations,
        continue_training=args.continue_training
    )
    
    print(f"Training completed. Results: {training_manager.get_training_summary()}")


def setup_reconstruction(args):
    """Set up and run reconstruction."""
    print("Setting up reconstruction...")
    
    config = create_default_config(args.config if hasattr(args, 'config') else None)
    
    # Create dataset
    dataset = PADataset(config)
    
    # Set up domain and forward model (simplified for CLI)
    from jwave.geometry import Domain
    from src.data.generation import setup_simulation_domain, create_wave_simulator
    
    domain, sensors, medium, time_axis = setup_simulation_domain(
        config.N, config.dx, config.sensor_margin, 
        config.pml_margin[0], config.num_sensors, 
        config.c, config.cfl, config.dims
    )
    
    forward_model = create_wave_simulator(sensors)
    
    # Create solver
    solver = create_solver(
        args.method, config, domain, forward_model
    )
    
    # Determine file range
    if args.files:
        if '-' in args.files:
            start, end = map(int, args.files.split('-'))
            file_indices = list(range(start, end + 1))
        else:
            file_indices = [int(args.files)]
    else:
        file_indices = list(range(config.recon_file_start, config.recon_file_end))
    
    # Run reconstruction
    results = batch_reconstruct(
        solver=solver,
        dataset=dataset,
        file_indices=file_indices,
        num_iterations=args.iterations,
        learning_rates=[config.lr_mu_r, config.lr_c_r],
        save_results=True
    )
    
    print(f"Reconstruction completed. Summary: {results['summary_stats']}")


def setup_visualization(args):
    """Set up visualization."""
    print("Setting up visualization...")
    
    config = create_default_config(args.config if hasattr(args, 'config') else None)
    
    if args.interactive:
        # Launch interactive visualization
        if config.dims == 3:
            from vis import VolumeVisualizer
            visualizer = VolumeVisualizer(config.data_path)
            visualizer.show()
        else:
            print("Starting Jupyter notebook for 2D visualization...")
            os.system("jupyter notebook vis.ipynb")
    else:
        # Create static plots
        setup_publication_style()
        
        dataset = PADataset(config)
        data = dataset[args.file_index]
        
        from src.visualization.plotting import plot_reconstruction_comparison
        
        # Load reconstruction results if available
        try:
            recon_data = dataset.load_reconstruction(args.file_index)
            
            plot_reconstruction_comparison(
                recon_data["mu_r"], recon_data["c_r"],
                data["mu"], data["c"],
                save_path=f"reconstruction_comparison_{args.file_index}.png"
            )
            print(f"Saved comparison plot for file {args.file_index}")
        except Exception as e:
            print(f"Could not create comparison plot: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PACT Reconstruction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate --config configs/default.yaml --batch-size 100
  %(prog)s train --model treenet --illuminations 5 --iterations 10
  %(prog)s reconstruct --method gradient_descent --files 0-10
  %(prog)s visualize --interactive --file-index 5
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (default: params.yaml)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data generation command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Number of datasets to generate"
    )
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train regularization networks")
    train_parser.add_argument(
        "--model", 
        type=str, 
        default="ynet",
        choices=["treenet", "treenet_p0", "ynet", "concatnet", "stepnet", "regnet"],
        help="Network architecture to train"
    )
    train_parser.add_argument(
        "--features", 
        type=int, 
        default=32,
        help="Number of base features"
    )
    train_parser.add_argument(
        "--illuminations", 
        type=int, 
        default=10,
        help="Number of illumination angles"
    )
    train_parser.add_argument(
        "--iterations", 
        type=int, 
        help="Number of training iterations"
    )
    train_parser.add_argument(
        "--continue", 
        dest="continue_training",
        action="store_true",
        help="Continue training from checkpoint"
    )
    
    # Reconstruction command
    recon_parser = subparsers.add_parser("reconstruct", help="Perform reconstruction")
    recon_parser.add_argument(
        "--method", 
        type=str, 
        default="gradient_descent",
        choices=["gradient_descent", "learned_regularization", "multi_parameter"],
        help="Reconstruction method"
    )
    recon_parser.add_argument(
        "--files", 
        type=str, 
        help="File indices to reconstruct (e.g., '5' or '0-10')"
    )
    recon_parser.add_argument(
        "--illuminations", 
        type=int, 
        default=10,
        help="Number of illumination angles"
    )
    recon_parser.add_argument(
        "--iterations", 
        type=int, 
        help="Number of reconstruction iterations"
    )
    recon_parser.add_argument(
        "--output", 
        type=str, 
        help="Output directory for results"
    )
    
    # Visualization command
    vis_parser = subparsers.add_parser("visualize", help="Visualize results")
    vis_parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Launch interactive visualization"
    )
    vis_parser.add_argument(
        "--file-index", 
        type=int, 
        default=0,
        help="File index to visualize"
    )
    vis_parser.add_argument(
        "--save-plots", 
        action="store_true",
        help="Save plots to disk"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Execute command
    try:
        if args.command == "generate":
            setup_data_generation(args)
        elif args.command == "train":
            setup_training(args)
        elif args.command == "reconstruct":
            setup_reconstruction(args)
        elif args.command == "visualize":
            setup_visualization(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()