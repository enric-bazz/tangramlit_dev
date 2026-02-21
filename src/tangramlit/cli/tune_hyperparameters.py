"""CLI wrapper to run Optuna tuning.

Accepts paths to scRNA dataset, spatial dataset, config file, and output directory.

Outputs written to the specified output directory:
  {output_dir}/{study_name}_best_params.yaml
  {output_dir}/{study_name}_trials.csv
  {output_dir}/{study_name}.db   # Optuna sqlite DB (if storage not provided)
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
        "tune", 
        description="Run Optuna tuning for hyperparameters optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Example:
        tangramlit tune \\
        -i /path/to/input_dir \\
        -sc scRNAseq_data.h5ad \\
        -ist IST_data.h5ad \\
        -c config.yaml \\
        -o /path/to/output_dir \\
        --n-trials 100 \\
        --storage sqlite:////custom/storage/study.db
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
    parser.add_argument('--study-name', default='tangram_optuna_study', help='Optuna study name (default: tangram_optuna_study)')
    parser.add_argument('--storage', default=None, help='Optuna storage URL (e.g. sqlite:///path/to.db). If omitted, sqlite DB is created in output_dir')
    parser.add_argument('--n-trials', type=int, default=40, help='Number of trials (default: 40)')
    parser.add_argument('--timeout', type=float, default=None, help='Timeout in seconds (default: None)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing study')
    parser.add_argument('--sampler', choices=['tpe', 'random'], default='tpe', help='Sampler type (default: tpe)')
    parser.add_argument('--dry-run', action='store_true', help='Only check files and config, do not run tuning')
    
    parser.set_defaults(func=run)


def run(args):
    import sys
    import yaml
    import scanpy as sc
    from pathlib import Path

    from ..hpo import tune_loss_coefficients
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

    # Hardcode config overrides
    config['lambda_sparsity_g1'] = 0
    config['num_epochs'] = 200

    random_state = config.get('random_state', None)

    # Load AnnData objects for tuning (read once)
    adata_sc = sc.read_h5ad(input_dir / args.scrna_name)
    adata_st = sc.read_h5ad(input_dir / args.ist_name)

    # Quick split genes to get train/val lists
    train_genes, val_genes, _ = split_train_val_test_genes(
        adata_sc, 
        adata_st, 
        random_state=random_state
    )

    # Prepare storage
    if args.storage is None:
        study_db = os.path.join(output_dir, f"{args.study_name}.db")
        storage = f"sqlite:///{study_db}"
    else:
        storage = args.storage

    print(f"Optuna storage: {storage}")

    if args.dry_run:
        print("Dry run: validation complete. Exiting.")
        return

    # Call tuning
    best_value, best_params, trials_df = tune_loss_coefficients(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=None,
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        **config,
        study_name=args.study_name,
        storage=storage,
        n_trials=args.n_trials,
        timeout=args.timeout,
        resume=args.resume,
        sampler_type=args.sampler,
    )

    # Save results
    out_best = os.path.join(output_dir, f"{args.study_name}_best_params.yaml")
    out_trials = os.path.join(output_dir, f"{args.study_name}_trials.csv")
    with open(out_best, 'w') as f:
        yaml.dump({'best_value': best_value, 'best_params': best_params}, f)
    trials_df.to_csv(out_trials, index=False)

    print(f"Saved best params to: {out_best}")
    print(f"Saved trials dataframe to: {out_trials}")