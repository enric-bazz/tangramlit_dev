"""CLI wrapper to train a spatial mapping model.

Accepts paths to scRNA dataset, spatial dataset, config file, and output directory.
Trains a Tangram mapper and saves results.

Outputs written to the specified output directory:
  {output_dir}/adata_maparser.h5ad          # Mapped single-cell data
  {output_dir}/training_results.yaml   # Training metadata
  {output_dir}/validation_results.yaml # (optional) Validation metrics
  {output_dir}/spatial_gene_comparison.csv  # (optional) Validation data
"""

import os
import argparse
from typing import Sequence


def validate_input_files(input_dir: str, file_names: Sequence[str]) -> None:
    """Ensure input files exist in the given directory."""
    missing = []
    for name in file_names:
        path = os.path.join(input_dir, name)
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))


def register(subparsers):
    parser = subparsers.add_parser(
        "train",  # name
        description="Train a spatial mapping model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:
            tangramlit train \\
            -i /path/to/input_dir \\
            -sc scRNAseq_data.h5ad \\
            -ist IST_data.h5ad \\
            -c config.yaml \\
            -o /path/to/output_dir \\
            --validate \\
            --device cuda
        """
    )
    # Mandatory arguments
    parser.add_argument('--input-dir', '-i',
                        required=True,
                        help="Path to input directory containing: \n " \
                            "a scRNA-seq h5ad file \n " \
                            "an IST h5ad file \n " \
                            "a configuration yaml file")
    parser.add_argument('--scrna-name', '-sc', 
                        required=True, 
                        help='Name of scRNA-seq h5ad file')
    parser.add_argument('--ist-name', '-ist', 
                        required=True, 
                        help='Name of IST h5ad file')
    parser.add_argument('--config-name', '-c', 
                        required=True, 
                        help='Name of mapping configuration yaml file')
    parser.add_argument('--output-dir', '-o', 
                        required=True, 
                        help='Path to output directory (will be created if not exists)')
    
    # Optional arguments
    parser.add_argument('--validate', action='store_true', help='Run validation and generate validation plots')
    parser.add_argument('--device', default='auto', help='Device to use for training (cpu, cuda, or auto)')
    parser.add_argument('--dry-run', action='store_true', help='Only check files and config, do not train')
    
    parser.set_defaults(func=run)


def run(args):
    import sys
    import yaml
    import scanpy as sc
    from pathlib import Path

    from ..mapping.trainer import map_cells_to_space
    from ..mapping.utils import validate_mapping_experiment, project_sc_genes_onto_space, compare_spatial_geneexp
    from ..data import split_train_val_test_genes

    # Convert to absolute paths
    input_dir = Path(os.path.abspath(args.input_dir))
    output_dir = Path(os.path.abspath(args.output_dir))

    # Validate input files
    try:
        validate_input_files(input_dir, 
                             [args.scrna_name, args.ist_name, args.config_name])
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load configuration
    with open(input_dir / args.config_name, 'r') as f:
        config = yaml.safe_load(f)

    # Handle config preprocessing
    if config.get('filter') and config.get('target_count') == 'None':
        config['target_count'] = None  # remove string formatting

    random_state = config.get('random_state', None)

    # Load data
    print("Loading data...")
    adata_sc = sc.read_h5ad(input_dir / args.scrna_name)
    adata_st = sc.read_h5ad(input_dir / args.ist_name)

    # Quick split genes to get train/val lists
    train_genes, val_genes, _ = split_train_val_test_genes(
        adata_sc,
        adata_st,
        random_state=random_state
    )

    print(f"Train genes: {len(train_genes)}, Val genes: {len(val_genes)}")

    if args.dry_run:
        print("Dry run: validation complete. Exiting.")
        return

    # Train mapper
    print("Training mapper...")
    adata_map, mapper, datamodule = map_cells_to_space(
        adata_sc=adata_sc,
        adata_st=adata_st,
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        device=args.device,
        **config
    )

    # Save results
    print("Saving results...")
    out_adata_map = os.path.join(output_dir, "adata_map.h5ad")
    adata_map.write_h5ad(out_adata_map)
    print(f"Saved mapped data to: {out_adata_map}")

    # Save training metadata
    out_metadata = os.path.join(output_dir, "training_results.yaml")
    metadata = {
        'n_train_genes': len(train_genes),
        'n_val_genes': len(val_genes),
        'mapper_type': mapper.__class__.__name__,
        'config': config,
    }
    with open(out_metadata, 'w') as f:
        yaml.dump(metadata, f)
    print(f"Saved training metadata to: {out_metadata}")

    # Validate if requested
    if args.validate:
        print("Running validation...")
        try:
            results = validate_mapping_experiment(mapper, datamodule)
            
            # Save validation results
            out_validation = os.path.join(output_dir, "validation_results.yaml")
            with open(out_validation, 'w') as f:
                yaml.dump(results, f)
            print(f"Saved validation results to: {out_validation}")
            
            # Generate plots
            print("Generating plots...")
            ad_ge = project_sc_genes_onto_space(adata_map, datamodule)
            df = compare_spatial_geneexp(ad_ge, datamodule)
            
            # Save plot data
            out_plot_data = os.path.join(output_dir, "spatial_gene_comparison.csv")
            df.to_csv(out_plot_data)
            print(f"Saved plot data to: {out_plot_data}")
            
        except Exception as e:
            print(f"Warning: Validation failed: {e}", file=sys.stderr)

    print("Training completed.")

# TODO: modify so that the preprocessed anndata (scrna and ist) are optionally written into h5ad