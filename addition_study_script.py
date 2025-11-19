"""Addition study runner CLI.

This script does the opposite of ablation_study_script.py. It starts with a baseline model
and adds individual terms from the ablation_schedule.json to study their isolated effects.

This script expects a dataset folder structured like the `optuna_tune_cli.py` expects:

  /.../data/DatasetX/
      scRNA_data.h5ad
      spatial_data.h5ad
      train_config.yaml
      ablation_schedule.json

It also expects that an Optuna best params YAML was previously saved into
  <root>/optuna_tests/DatasetX/{study_name}_best_params.yaml

The script will:
 - load train_config.yaml and best params YAML, merge them (best params override)
 - read `ablation_schedule.json` (list of parameter names)
 - run experiments: baseline, vanilla, individual term additions, and full model
 - for each configuration, call `tangramlit.map_cells_to_space` and run benchmarking
 - save results to `<root>/addition_study/DatasetX/{study_name}/`
 - write a CSV with validation results summary

"""

import argparse
import os
import sys
import gc
import csv
import json
import yaml
import shutil
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import tangramlit as tgl


def validate_dataset_folder(dataset_folder: str):
    sc_name = 'scRNA_data.h5ad'
    st_name = 'spatial_data.h5ad'
    cfg_name = 'train_config.yaml'

    sc_path = os.path.join(dataset_folder, sc_name)
    st_path = os.path.join(dataset_folder, st_name)
    cfg_path = os.path.join(dataset_folder, cfg_name)

    return sc_path, st_path, cfg_path


def infer_output_root(dataset_folder: str) -> str:
    parent = os.path.dirname(dataset_folder)
    parent_basename = os.path.basename(parent)
    if parent_basename == 'data':
        return os.path.dirname(parent)
    return parent


def read_ablation_schedule(schedule_path: str) -> List:
    """Read ablation schedule from JSON.

    The JSON file is expected to contain a list. Each element can be a string
    (single parameter name) or a list of strings (multiple parameters to ablate
    together).
    """
    with open(schedule_path, 'r') as f:
        data = json.load(f)

    return data


def check_ablation_results_exist(output_root: str, dataset_name: str) -> dict:
    """Check if baseline, vanilla, full results exist in ablation_study.
    
    Returns dict with keys 'baseline', 'vanilla', 'full' and boolean values.
    """
    ablation_dir = os.path.join(output_root, 'ablation_study', dataset_name)
    
    results = {'baseline': False, 'vanilla': False, 'full': False}
    
    if os.path.exists(ablation_dir):
        # Check for ablated_baseline, ablated_vanilla, ablated_full directories
        for config_type in ['baseline', 'vanilla', 'full']:
            ablated_dir = os.path.join(ablation_dir, f'ablated_{config_type}')
            benchmark_file = os.path.join(ablated_dir, 'benchmark.xlsx')
            h5ad_file = os.path.join(ablation_dir, f'{dataset_name}_{config_type}.h5ad')
            
            if os.path.exists(benchmark_file) and os.path.exists(h5ad_file):
                results[config_type] = True
                print(f"Found existing {config_type} results in ablation_study for {dataset_name}")
    
    return results


def copy_ablation_results(output_root: str, dataset_name: str, config_name: str, addition_outdir: str):
    """Copy results from ablation_study to addition_study.
    
    config_name should be 'baseline', 'vanilla', or 'full'
    """
    ablation_dir = os.path.join(output_root, 'ablation_study', dataset_name)
    
    # Copy the h5ad file
    src_h5ad = os.path.join(ablation_dir, f'{dataset_name}_{config_name}.h5ad')
    dst_h5ad = os.path.join(addition_outdir, f'{dataset_name}_{config_name}.h5ad')
    
    if os.path.exists(src_h5ad):
        import shutil
        shutil.copy2(src_h5ad, dst_h5ad)
        print(f"Copied {config_name} h5ad: {src_h5ad} -> {dst_h5ad}")
    
    # Copy the entire ablated directory
    src_dir = os.path.join(ablation_dir, f'ablated_{config_name}')
    dst_dir = os.path.join(addition_outdir, f'added_{config_name}')
    
    if os.path.exists(src_dir):
        import shutil
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print(f"Copied {config_name} results: {src_dir} -> {dst_dir}")
        return True
    
    return False


# Keys we will pass to map_cells_to_space from the merged config (whitelist)
_MAP_KEYS = [
    'filter', 'learning_rate', 'num_epochs', 'random_state',
    'lambda_d', 'lambda_g1', 'lambda_g2', 'lambda_r',
    'lambda_l1', 'lambda_l2', 'lambda_count', 'lambda_f_reg', 'target_count',
    'lambda_sparsity_g1', 'lambda_neighborhood_g1', 'lambda_getis_ord',
    'lambda_moran', 'lambda_geary', 'lambda_ct_islands', 'cluster_label', 'input_genes'
]

# Hardcoded baseline model terms (these are set to 0 in baseline, lambda_g1 is fixed to 1)
_BASELINE_TERMS = [
    'lambda_d', 'lambda_g2', 'lambda_r',
    'lambda_l1', 'lambda_l2',
    'lambda_sparsity_g1', 'lambda_neighborhood_g1', 'lambda_getis_ord', 
    'lambda_moran', 'lambda_geary', 'lambda_ct_islands'
]

# Hardcoded vanilla model terms (these are set to 0 in vanilla)
_VANILLA_TERMS = [
    'lambda_l1', 'lambda_l2', 'lambda_sparsity_g1', 
    'lambda_neighborhood_g1', 'lambda_getis_ord', 'lambda_moran', 'lambda_geary', 
    'lambda_ct_islands'
]


def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="Run addition study on Dataset folder")
    p.add_argument('dataset_folder', help='Path to Dataset folder containing scRNA_data.h5ad, spatial_data.h5ad, train_config.yaml')
    p.add_argument('--study-name', default='tangram_optuna_study', help='Optuna study name used to write best params file')
    p.add_argument('--reuse-ablation', action='store_true', 
                   help='Skip generating baseline/vanilla/full if they exist in ablation_study')
    args = p.parse_args(argv)

    dataset_folder = os.path.abspath(args.dataset_folder)
    sc_path, st_path, cfg_path = validate_dataset_folder(dataset_folder)
    dataset_name = os.path.basename(dataset_folder.rstrip(os.sep))

    # load train config
    with open(cfg_path, 'r') as f:
        train_config = yaml.safe_load(f)

    # infer output root and locate optuna best params
    output_root = infer_output_root(dataset_folder)
    optuna_outdir = os.path.join(output_root, 'optuna_tests', dataset_name)
    best_params_path = os.path.join(optuna_outdir, f"{args.study_name}_best_params.yaml")

    with open(best_params_path, 'r') as f:
        bp = yaml.safe_load(f)
        best_params = bp['best_params']
    print(f"Loaded best params from: {best_params_path}")

    # Merge: best_params override train_config keys when present
    merged_base = dict(train_config)
    merged_base.update(best_params)

    # read ablation schedule (JSON list)
    schedule_path = os.path.join(dataset_folder, 'ablation_schedule.json')
    schedule = read_ablation_schedule(schedule_path)
    print(f"Addition schedule entries: {schedule}")

    # Quick split of training/validation genes (used for map call). Reuse tangramlit helper
    # read the AnnData objects once (map_cells_to_space will receive AnnData objects)
    adata_sc = sc.read_h5ad(sc_path)
    adata_st = sc.read_h5ad(st_path)

    train_genes, val_genes = tgl.split_train_val_genes(adata_sc, adata_st, random_state=merged_base.get('random_state', None))

    # prepare addition output dir
    addition_outdir = os.path.join(output_root, 'addition_study', dataset_name, args.study_name)
    os.makedirs(addition_outdir, exist_ok=True)

    results = []

    # Create individual addition configurations
    # Start with baseline (all baseline terms set to 0, lambda_g1 fixed to 1)
    # Then add each term from the schedule individually to study isolated effects
    
    # Convert schedule to flat list of individual terms
    flat_schedule = []
    for term in schedule:
        if isinstance(term, str):
            flat_schedule.append(term)
        else:
            flat_schedule.extend(term)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_schedule = []
    for term in flat_schedule:
        if term not in seen:
            seen.add(term)
            unique_schedule.append(term)
    
    print(f"Individual addition sequence: {unique_schedule}")

    # Check if we should reuse ablation results
    existing_ablation = {} 
    if args.reuse_ablation:
        existing_ablation = check_ablation_results_exist(output_root, dataset_name)
    
    # Generate configurations: baseline, vanilla, then each term individually, then full
    configurations = []
    
    # 1. Baseline configuration (all baseline terms set to 0)
    if not existing_ablation.get('baseline', False):
        configurations.append(('baseline', []))
    
    # 2. Vanilla configuration (only vanilla terms set to 0)
    if not existing_ablation.get('vanilla', False):
        configurations.append(('vanilla', 'vanilla_special'))
    
    # 3. Individual addition configurations (baseline + one term each)
    for term in unique_schedule:
        config_name = f"add_{term}"
        configurations.append((config_name, [term]))
    
    # 4. Full configuration (all terms active from optuna best params)
    if not existing_ablation.get('full', False):
        configurations.append(('full', _BASELINE_TERMS))

    # First, copy any existing ablation results if reuse is enabled
    if args.reuse_ablation:
        for config_type in ['baseline', 'vanilla', 'full']:
            if existing_ablation.get(config_type, False):
                if copy_ablation_results(output_root, dataset_name, config_type, addition_outdir):
                    # Read the validation results from the copied CSV if available
                    ablation_csv = os.path.join(output_root, 'ablation_study', dataset_name, f'{dataset_name}_validation_results.csv')
                    if os.path.exists(ablation_csv):
                        try:
                            df_ablation = pd.read_csv(ablation_csv)
                            # Find the row for this configuration
                            config_row = df_ablation[df_ablation['ablated_term'] == config_type]
                            if not config_row.empty:
                                row_data = config_row.iloc[0]
                                added_terms_str = config_type
                                num_terms = 0 if config_type in ['baseline', 'vanilla'] else len(_BASELINE_TERMS) if config_type == 'full' else 1
                                
                                record = {
                                    'configuration': config_type,
                                    'added_terms': added_terms_str,
                                    'num_added_terms': num_terms,
                                    'auc_train': float(row_data.get('auc_train', 0)),
                                    'auc_val': float(row_data.get('auc_val', 0)),
                                    'corr_train_moran': float(row_data.get('corr_train_moran', 0)),
                                    'corr_val_moran': float(row_data.get('corr_val_moran', 0)),
                                    'corr_train_geary': float(row_data.get('corr_train_geary', 0)),
                                    'corr_val_geary': float(row_data.get('corr_val_geary', 0)),
                                }
                                
                                # Add mean metrics if available
                                metrics = ["score", "SSIM", "PCC", "RMSE", "JS"]
                                for m in metrics:
                                    for split in ['train', 'val']:
                                        key = f'mean_{split}_{m}'
                                        record[key] = float(row_data.get(key, 0))
                                
                                results.append(record)
                                print(f"Added existing {config_type} results to summary")
                        except Exception as e:
                            print(f"Warning: Could not read ablation results for {config_type}: {e}")
    
    for config_name, added_terms in configurations:
        print(f"Running addition study for configuration: {config_name}")
        
        # copy merged base and configure according to model type
        run_config = dict(merged_base)
        
        if added_terms == 'vanilla_special':
            # Vanilla model: only set vanilla terms to 0, keep others active
            for t in _VANILLA_TERMS:
                run_config[t] = 0
        elif config_name == 'full':
            # Full model: restore all baseline terms from best params
            for t in added_terms:
                if t in merged_base:
                    run_config[t] = merged_base[t]
        else:
            # Baseline or individual addition: start with baseline (all baseline terms to 0)
            for t in _BASELINE_TERMS:
                if t != 'lambda_g1':  # lambda_g1 is fixed and not modified
                    run_config[t] = 0
            
            # Add back the terms we want to include in this configuration
            for t in added_terms:
                if t in merged_base:
                    run_config[t] = merged_base[t]  # Restore original value

        # Build kwargs for map_cells_to_space using whitelist
        map_kwargs = {}
        for k in _MAP_KEYS:
            if k in run_config:
                map_kwargs[k] = run_config[k]
        
        adata_map, model, datamodule = tgl.map_cells_to_space(
            adata_sc=adata_sc,
            adata_st=adata_st,
            input_genes=map_kwargs.pop('input_genes', None),
            train_genes_names=train_genes,
            val_genes_names=val_genes,
            experiment_name=f"{dataset_name}_{config_name}",
            **map_kwargs,
        )
        
        # save the produced AnnData
        # out_h5ad = os.path.join(addition_outdir, f"{dataset_name}_{config_name}.h5ad")
        # adata_map.write_h5ad(out_h5ad)
        # print(f"Saved mapped AnnData to: {out_h5ad}")

        # Run benchmarking and gene-level analyses
        # create iteration-specific output folder
        iter_outdir = os.path.join(addition_outdir, f"added_{config_name}")
        os.makedirs(iter_outdir, exist_ok=True)

        # Project sc genes onto space, run the unified benchmark on the projected
        # spatial expression and save outputs (benchmark.xlsx).
        adata_ge = tgl.project_sc_genes_onto_space(adata_map, datamodule)
        # New benchmark_mapping expects projected AnnData and returns a per-gene DataFrame
        df_g = tgl.benchmark_mapping(adata_ge, datamodule)

        # Save benchmark DataFrame as Excel workbook
        bench_xlsx = os.path.join(iter_outdir, "benchmark.xlsx")
        df_g.to_excel(bench_xlsx, index=True)

        # Produce AUC plot from per-gene dataframe
        (fig_train, fig_val), (auc_train, auc_val) = tgl.plot_auc_curve(df_g, plot_train=True, plot_validation=True)
        for fig, name in zip((fig_train, fig_val), ("train", "val")):
            fig_path = os.path.join(iter_outdir, f"auc_plot_{name}.png")
            fig.savefig(fig_path)
            plt.close(fig)

        # Plot score vs spatial-autocorrelation statistics and save figures
        sac_figs, sac_corrs = tgl.plot_score_SA_corr(df_g, plot_train=True, plot_validation=True, plot_fit=True)
        sac_names = ("sac_train_moran", "sac_val_moran", "sac_train_geary", "sac_val_geary")
        for fig, name in zip(sac_figs, sac_names):
            fig_path = os.path.join(iter_outdir, f"{name}.png")
            fig.savefig(fig_path)
            plt.close(fig)

        # Plot and save score histograms for train/validation
        hist_figs = tgl.plot_score_histograms(df_g, bins=20, alpha=0.8, plot_train=True, plot_validation=True)
        hist_names = ("hist_train_scores", "hist_val_scores")
        for fig, name in zip(hist_figs, hist_names):
            fig_path = os.path.join(iter_outdir, f"{name}.png")
            fig.savefig(fig_path)
            plt.close(fig)

        # Unpack correlations
        (corr_train_moran, corr_val_moran, corr_train_geary, corr_val_geary) = sac_corrs

        # Compute mean metrics separately for training and validation genes
        # using the DataFrame returned by benchmark_mapping (df_g).
        metrics = ["score", "SSIM", "PCC", "RMSE", "JS"]
        train_mask = df_g["is_training"].astype(bool)
        val_mask = df_g["is_validation"].astype(bool)

        metric_means = {}
        for m in metrics:
            metric_means[f"mean_train_{m}"] = float(np.nanmean(df_g.loc[train_mask, m].to_numpy()))
            metric_means[f"mean_val_{m}"] = float(np.nanmean(df_g.loc[val_mask, m].to_numpy()))

        # add a summary record for this addition using df_g-derived metrics
        added_terms_str = 'vanilla' if added_terms == 'vanilla_special' else ('+'.join(added_terms) if added_terms else 'none')
        num_terms = 0 if added_terms == 'vanilla_special' or not added_terms else len(added_terms)
        
        record = {
            'configuration': config_name,
            'added_terms': added_terms_str,
            'num_added_terms': num_terms,
            'auc_train': float(auc_train),
            'auc_val': float(auc_val),
            'corr_train_moran': float(corr_train_moran),
            'corr_val_moran': float(corr_val_moran),
            'corr_train_geary': float(corr_train_geary),
            'corr_val_geary': float(corr_val_geary),
        }
        record.update(metric_means)
        results.append(record)

        # clean for memory build up
        del adata_map, model, datamodule
        gc.collect()

    # write results csv
    df = pd.DataFrame(results)
    out_csv = os.path.join(addition_outdir, f"{dataset_name}_addition_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved addition study results to: {out_csv}")


if __name__ == '__main__':
    main()
