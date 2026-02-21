"""CLI wrapper for cross-validation of spatial mapping across gene sets.

Accepts paths to scRNA dataset, spatial dataset, config file, and output directory.
Performs k-fold cross-validation and saves results.

Outputs written to the specified output directory:
  {output_dir}/cv_metrics.yaml        # Average metrics across folds
  {output_dir}/cv_metrics_per_fold.csv  # Per-fold metrics
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
        "cv",  # name
        description="Cross-validate spatial mapping across gene sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        tangramlit cv \\
        -i /path/to/input_dir \\
        -sc scRNAseq_data.h5ad \\
        -ist IST_data.h5ad \\
        -c config.yaml \\
        -o /path/to/output_dir \\
        --k-folds 10 \\
        --dry-run
        """
    )
    
    # Mandatory positional arguments
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
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--dry-run', action='store_true', help='Only check files and config, do not run cross-validation')
    
    parser.set_defaults(func=run)


def  run(args):
    import sys
    import yaml
    import pandas as pd
    import scanpy as sc
    from pathlib import Path

    from ..data import kfold_gene_splits
    from ..mapping.trainer import map_cells_to_space
    from ..mapping.utils import validate_mapping_experiment, project_sc_genes_onto_space, compare_spatial_geneexp
    from ..benchmarking import benchmark_mapping, aggregate_benchmarking_metrics

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

    # Bring gene names to lowercase
    adata_sc.var_names = adata_sc.var_names.str.lower()
    adata_st.var_names = adata_st.var_names.str.lower()

    # Get shared genes
    input_genes = sorted(set(adata_sc.var_names) & set(adata_st.var_names))
    print(f"Shared genes: {len(input_genes)}")
    print(f"K-folds: {args.k_folds}")

    if args.dry_run:
        print("Dry run: validation complete. Exiting.")
        return

    # Perform cross-validation
    print("Running cross-validation...")
    cv_metrics = {}
    fold_metrics_per_fold = []
    
    for fold_idx, (train_genes, test_genes) in enumerate(kfold_gene_splits(genes=input_genes, k=args.k_folds, random_state=random_state), 1):
        print(f"  Fold {fold_idx}/{args.k_folds}: {len(train_genes)} train genes, {len(test_genes)} test genes")
        
        # Train mapper for this fold
        adata_map, mapper, datamodule = map_cells_to_space(
            adata_sc=adata_sc,
            adata_st=adata_st,
            input_genes=input_genes,
            train_genes_names=train_genes,
            val_genes_names=test_genes,
            **config
        )

        # Compute fold metrics (internal and benchmarking)
        fold_results = validate_mapping_experiment(mapper, datamodule)[0]  # extract dict from list 
        adata_ge = project_sc_genes_onto_space(adata_map, datamodule)
        fold_df_g = compare_spatial_geneexp(adata_ge, datamodule)
        fold_df_g = benchmark_mapping(adata_ge, datamodule, fold_df_g)
        fold_summary = aggregate_benchmarking_metrics(fold_df_g)

        print(f"    Fold {fold_idx} completed.")
        
        # Accumulate metrics
        fold_metrics_per_fold.append({'fold': fold_idx, **fold_results, **fold_summary['mean']})
        for key, value in fold_results.items():
            if key not in cv_metrics:
                cv_metrics[key] = []
            cv_metrics[key].append(value)
        # Add benchmarking metrics mean
        for ind in fold_summary.index:
            if ind not in cv_metrics:
                cv_metrics[ind] = []
            cv_metrics[ind] = fold_summary['mean'].loc[ind]

    # Calculate average metrics
    print("Computing average metrics...")
    avg_metrics = {}
    for key, values in cv_metrics.items():
        avg_metrics[key] = {
            'mean': float(pd.Series(values).mean()),
            'std': float(pd.Series(values).std()),
            'min': float(pd.Series(values).min()),
            'max': float(pd.Series(values).max()),
        }

    # Save results
    print("Saving results...")
    
    # Save average metrics
    out_avg_metrics = os.path.join(output_dir, "cv_metrics.yaml")
    with open(out_avg_metrics, 'w') as f:
        yaml.dump(avg_metrics, f)
    print(f"Saved average metrics to: {out_avg_metrics}")

    # Save per-fold metrics
    out_per_fold = os.path.join(output_dir, "cv_metrics_per_fold.csv")
    df_per_fold = pd.DataFrame(fold_metrics_per_fold)
    df_per_fold.to_csv(out_per_fold, index=False)
    print(f"Saved per-fold metrics to: {out_per_fold}")

    print("Cross-validation completed.")