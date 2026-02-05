"""Fast ablation script for lambda_d term on all datasets.

This script runs ablation of lambda_d on all datasets found in the data/ folder,
updating the validation results CSV and creating the ablated_lambda_d folder.
"""

import argparse
import os
import gc
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import tangramlit as tgl


def find_datasets(data_root: str):
    """Find all dataset folders in data_root."""
    datasets = []
    if not os.path.exists(data_root):
        return datasets
    
    for item in os.listdir(data_root):
        dataset_path = os.path.join(data_root, item)
        if os.path.isdir(dataset_path):
            # Check if it has the required files
            sc_path = os.path.join(dataset_path, 'scRNA_data.h5ad')
            st_path = os.path.join(dataset_path, 'spatial_data.h5ad')
            cfg_path = os.path.join(dataset_path, 'train_config.yaml')
            if os.path.exists(sc_path) and os.path.exists(st_path) and os.path.exists(cfg_path):
                datasets.append((item, dataset_path))
    
    return datasets


_MAP_KEYS = [
    'filter', 'learning_rate', 'num_epochs', 'random_state',
    'lambda_d', 'lambda_g1', 'lambda_g2', 'lambda_r',
    'lambda_l1', 'lambda_l2', 'lambda_count', 'lambda_f_reg', 'target_count',
    'lambda_sparsity_g1', 'lambda_neighborhood_g1', 'lambda_getis_ord',
    'lambda_moran', 'lambda_geary', 'lambda_ct_islands', 'cluster_label', 'input_genes'
]


def process_dataset(dataset_name: str, dataset_path: str, study_name: str, output_root: str):
    """Process a single dataset with lambda_d ablation."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Paths
    sc_path = os.path.join(dataset_path, 'scRNA_data.h5ad')
    st_path = os.path.join(dataset_path, 'spatial_data.h5ad')
    cfg_path = os.path.join(dataset_path, 'train_config.yaml')
    
    # Load train config
    with open(cfg_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # Load best params
    optuna_outdir = os.path.join(output_root, 'optuna_tests', dataset_name)
    best_params_path = os.path.join(optuna_outdir, f"{study_name}_best_params.yaml")
    
    if not os.path.exists(best_params_path):
        print(f"WARNING: Best params not found at {best_params_path}, skipping dataset")
        return None
    
    with open(best_params_path, 'r') as f:
        bp = yaml.safe_load(f)
        best_params = bp['best_params']
    print(f"Loaded best params from: {best_params_path}")
    
    # Merge configs
    merged_base = dict(train_config)
    merged_base.update(best_params)
    
    # Load data
    adata_sc = sc.read_h5ad(sc_path)
    adata_st = sc.read_h5ad(st_path)
    
    # Split genes
    train_genes, val_genes = tgl.split_train_val_genes(
        adata_sc, adata_st, random_state=merged_base.get('random_state', None)
    )
    
    # Prepare output directory
    ablation_outdir = os.path.join(output_root, 'ablation_study', dataset_name) #  , study_name)
    os.makedirs(ablation_outdir, exist_ok=True)
    
    # Create config with lambda_d = 0
    run_config = dict(merged_base)
    run_config['lambda_d'] = 0
    
    # Build map kwargs
    map_kwargs = {}
    for k in _MAP_KEYS:
        if k in run_config:
            map_kwargs[k] = run_config[k]
    
    ablated_term = 'lambda_d'
    print(f"Running ablation for term: {ablated_term}")
    
    # Map cells
    adata_map, model, datamodule = tgl.map_cells_to_space(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=map_kwargs.pop('input_genes', None),
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        experiment_name=f"{dataset_name}_{ablated_term}",
        **map_kwargs,
    )
    
    # Save mapped data
    out_h5ad = os.path.join(ablation_outdir, f"{dataset_name}_{ablated_term}.h5ad")
    adata_map.write_h5ad(out_h5ad)
    print(f"Saved mapped AnnData to: {out_h5ad}")
    
    # Create iteration output folder
    iter_outdir = os.path.join(ablation_outdir, f"ablated_{ablated_term}")
    os.makedirs(iter_outdir, exist_ok=True)
    
    # Project genes and benchmark
    adata_ge = tgl.project_sc_genes_onto_space(adata_map, datamodule)
    df_g = tgl.benchmark_mapping(adata_ge, datamodule)
    
    # Save benchmark
    bench_xlsx = os.path.join(iter_outdir, "benchmark.xlsx")
    df_g.to_excel(bench_xlsx, index=True)
    print(f"Saved benchmark to: {bench_xlsx}")
    
    # AUC plots
    (fig_train, fig_val), (auc_train, auc_val) = tgl.plot_auc_curve(
        df_g, plot_train=True, plot_validation=True
    )
    for fig, name in zip((fig_train, fig_val), ("train", "val")):
        fig_path = os.path.join(iter_outdir, f"auc_plot_{name}.png")
        fig.savefig(fig_path)
        plt.close(fig)
    
    # SAC plots
    sac_figs, sac_corrs = tgl.plot_score_SA_corr(
        df_g, plot_train=True, plot_validation=True, plot_fit=True
    )
    sac_names = ("sac_train_moran", "sac_val_moran", "sac_train_geary", "sac_val_geary")
    for fig, name in zip(sac_figs, sac_names):
        fig_path = os.path.join(iter_outdir, f"{name}.png")
        fig.savefig(fig_path)
        plt.close(fig)
    
    # Score histograms
    hist_figs = tgl.plot_score_histograms(
        df_g, bins=20, alpha=0.8, plot_train=True, plot_validation=True
    )
    hist_names = ("hist_train_scores", "hist_val_scores")
    for fig, name in zip(hist_figs, hist_names):
        fig_path = os.path.join(iter_outdir, f"{name}.png")
        fig.savefig(fig_path)
        plt.close(fig)
    
    # Unpack correlations
    (corr_train_moran, corr_val_moran, corr_train_geary, corr_val_geary) = sac_corrs
    
    # Compute metrics
    metrics = ["score", "SSIM", "PCC", "RMSE", "JS"]
    train_mask = df_g["is_training"].astype(bool)
    val_mask = df_g["is_validation"].astype(bool)
    
    metric_means = {}
    for m in metrics:
        metric_means[f"mean_train_{m}"] = float(np.nanmean(df_g.loc[train_mask, m].to_numpy()))
        metric_means[f"mean_val_{m}"] = float(np.nanmean(df_g.loc[val_mask, m].to_numpy()))
    
    # Create record
    record = {
        'ablated_term': ablated_term,
        'auc_train': float(auc_train),
        'auc_val': float(auc_val),
        'corr_train_moran': float(corr_train_moran),
        'corr_val_moran': float(corr_val_moran),
        'corr_train_geary': float(corr_train_geary),
        'corr_val_geary': float(corr_val_geary),
    }
    record.update(metric_means)
    
    # Update validation results CSV
    csv_path = os.path.join(ablation_outdir, f"{dataset_name}_validation_results.csv")
    
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # Remove old lambda_d entry if exists
        df_existing = df_existing[df_existing['ablated_term'] != ablated_term]
        # Append new record
        df_updated = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([record])
    
    df_updated.to_csv(csv_path, index=False)
    print(f"Updated validation results: {csv_path}")
    
    # Cleanup
    del adata_map, model, datamodule, adata_sc, adata_st, adata_ge
    gc.collect()
    
    print(f"Completed dataset: {dataset_name}")
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Run lambda_d ablation on all datasets"
    )
    parser.add_argument(
        '--data-root',
        default='data',
        help='Root directory containing dataset folders (default: data)'
    )
    parser.add_argument(
        '--study-name',
        default='tangram_optuna_study',
        help='Optuna study name (default: tangram_optuna_study)'
    )
    parser.add_argument(
        '--output-root',
        default=None,
        help='Output root directory (default: parent of data-root)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_root = os.path.abspath(args.data_root)
    if args.output_root:
        output_root = os.path.abspath(args.output_root)
    else:
        output_root = os.path.dirname(data_root)
    
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Study name: {args.study_name}")
    
    # Find datasets
    datasets = find_datasets(data_root)
    print(f"\nFound {len(datasets)} datasets: {[name for name, _ in datasets]}")
    
    if not datasets:
        print("No datasets found. Exiting.")
        return
    
    # Process each dataset
    results = []
    for dataset_name, dataset_path in datasets:
        try:
            record = process_dataset(dataset_name, dataset_path, args.study_name, output_root)
            if record:
                record['dataset'] = dataset_name
                results.append(record)
        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Processed {len(results)} datasets successfully")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['dataset']}: AUC_val={r['auc_val']:.4f}, mean_val_score={r['mean_val_score']:.4f}")


if __name__ == '__main__':
    main()