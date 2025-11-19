"""Ablation study runner CLI.

This script expects a dataset folder structured like the `optuna_tune_cli.py` expects:

  /.../data/DatasetX/
      scRNA_data.h5ad
      spatial_data.h5ad
      train_config.yaml
      ablation_schedule.json

It also expects that an Optuna best params YAML was previously saved into
  <root>/optuna_tests/DatasetX/{study_name}_best_params.yaml

The script will:
 - load train_config.yaml
 - load best params YAML (if present) and merge keys into config (best params override train_config keys)
 - read `ablation_schedule.json` (list of parameter names)
 - for each parameter (or list of parameters) name in schedule, set that parameter value to 0 in the merged config,
   call `tangramlit.map_cells_to_space` and `tangramlit.validate_mapping_experiment`,
   save the produced AnnData to `<root>/ablation_study/DatasetX/` and accumulate validation results.
 - write a CSV with validation results to the same ablation output folder.

"""

import argparse
import os
import sys
import gc
import csv
import json
import yaml
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


# Keys we will pass to map_cells_to_space from the merged config (whitelist)
_MAP_KEYS = [
    'filter', 'learning_rate', 'num_epochs', 'random_state',
    'lambda_d', 'lambda_g1', 'lambda_g2', 'lambda_r',
    'lambda_l1', 'lambda_l2', 'lambda_count', 'lambda_f_reg', 'target_count',
    'lambda_sparsity_g1', 'lambda_neighborhood_g1', 'lambda_getis_ord',
    'lambda_moran', 'lambda_geary', 'lambda_ct_islands', 'cluster_label', 'input_genes'
]


def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="Run ablation study on Dataset folder")
    p.add_argument('dataset_folder', help='Path to Dataset folder containing scRNA_data.h5ad, spatial_data.h5ad, train_config.yaml')
    p.add_argument('--study-name', default='tangram_optuna_study', help='Optuna study name used to write best params file')
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
    print(f"Ablation schedule entries: {schedule}")

    # Quick split of training/validation genes (used for map call). Reuse tangramlit helper
    # read the AnnData objects once (map_cells_to_space will receive AnnData objects)
    adata_sc = sc.read_h5ad(sc_path)
    adata_st = sc.read_h5ad(st_path)

    train_genes, val_genes = tgl.split_train_val_genes(adata_sc, adata_st, random_state=merged_base.get('random_state', None))

    # prepare ablation output dir
    ablation_outdir = os.path.join(output_root, 'ablation_study', dataset_name, args.study_name)
    os.makedirs(ablation_outdir, exist_ok=True)

    results = []

    # add ful, baseline and vanilla models to schedule
    schedule.append([])
    schedule.append(['lambda_d', 'lambda_g2', 'lambda_r', 
                            'lambda_l1', 'lambda_l2', 'lambda_sparsity_g1', 
                            'lambda_neighborhood_g1', 'lambda_getis_ord', 'lambda_moran', 'lambda_geary', 
                            'lambda_ct_islands'])

    schedule.append(['lambda_l1', 'lambda_l2', 'lambda_sparsity_g1', 
                           'lambda_neighborhood_g1', 'lambda_getis_ord', 'lambda_moran', 'lambda_geary', 
                           'lambda_ct_islands'])

    for term in schedule:
        # term may be a string or a list of strings
        if isinstance(term, str):
            term_list = [term]
        else:
            term_list = list(term)

        print(f"Running ablation for terms: {term_list}")
        # copy merged base and set the ablated terms to 0
        run_config = dict(merged_base)
        for t in term_list:
            run_config[t] = 0

        # Build kwargs for map_cells_to_space using whitelist
        map_kwargs = {}
        for k in _MAP_KEYS:
            if k in run_config:
                map_kwargs[k] = run_config[k]
        # create a safe name for the experiment and output file by joining terms with '+' or recognising full, baseline and vanilla
        if len(term_list) == 0:
            joined_terms = 'full'
        elif len(term_list) == 11:
            joined_terms = 'baseline'
        elif len(term_list) == 8:
            joined_terms = 'vanilla'
        else:
            joined_terms = '+'.join(term_list)
        
        adata_map, model, datamodule = tgl.map_cells_to_space(
            adata_sc=adata_sc,
            adata_st=adata_st,
            input_genes=map_kwargs.pop('input_genes', None),
            train_genes_names=train_genes,
            val_genes_names=val_genes,
            experiment_name=f"{dataset_name}_{joined_terms}",
            **map_kwargs,
        )
        # save the produced AnnData
        out_h5ad = os.path.join(ablation_outdir, f"{dataset_name}_{joined_terms}.h5ad")
        adata_map.write_h5ad(out_h5ad)
        print(f"Saved mapped AnnData to: {out_h5ad}")

        # Instead of running validate_mapping_experiment, run benchmarking and gene-level analyses
        # create iteration-specific output folder
        iter_outdir = os.path.join(ablation_outdir, f"ablated_{joined_terms}")
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

        # add a summary record for this ablation using df_g-derived metrics
        record = {
            'ablated_term': joined_terms,
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
    out_csv = os.path.join(ablation_outdir, f"{dataset_name}_validation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved ablation validation results to: {out_csv}")


if __name__ == '__main__':
    main()
