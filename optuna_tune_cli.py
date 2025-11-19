"""CLI wrapper to run Optuna tuning for a Dataset folder.

Expected dataset structure (what the CLI expects at the dataset_folder path):

  /.../data/DatasetX/
      scRNA_data.h5ad
      spatial_data.h5ad
      train_config.yaml

The CLI takes the path to the Dataset folder (for example: `/.../data/Dataset3`).
It will extract `Dataset3` from that path and write Optuna outputs into
`<root>/optuna_tests/Dataset3/` where `<root>` is inferred as the parent of
the `data/` folder if present, otherwise as the parent of the provided
dataset folder.

Outputs written to:
  <root>/optuna_tests/DatasetX/{study_name}_best_params.yaml
  <root>/optuna_tests/DatasetX/{study_name}_trials.csv
  <root>/optuna_tests/DatasetX/{study_name}.db   # Optuna sqlite DB (if storage not provided)
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import yaml
import scanpy as sc
import tangramlit as tgl


def validate_dataset_folder(dataset_folder: str) -> Tuple[str, str, str]:
    """Ensure dataset_folder contains expected files and return their paths.

    Returns (sc_path, st_path, config_path).
    """
    sc_name = 'scRNA_data.h5ad'
    st_name = 'spatial_data.h5ad'
    cfg_name = 'train_config.yaml'

    sc_path = os.path.join(dataset_folder, sc_name)
    st_path = os.path.join(dataset_folder, st_name)
    cfg_path = os.path.join(dataset_folder, cfg_name)

    missing = [n for n, p in [(sc_name, sc_path), (st_name, st_path), (cfg_name, cfg_path)] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Dataset folder {dataset_folder} missing expected files: {missing}")
    return sc_path, st_path, cfg_path


def infer_output_root(dataset_folder: str) -> str:
    """Infer project root where to place `optuna_tests/DatasetX`.

    If dataset_folder is .../data/DatasetX, return parent of data (two levels up).
    Otherwise return parent of dataset_folder.
    """
    parent = os.path.dirname(dataset_folder)
    parent_basename = os.path.basename(parent)
    if parent_basename == 'data':
        return os.path.dirname(parent)
    return parent


def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="Run Optuna tuning for a Dataset folder")
    p.add_argument('dataset_folder', help='Path to Dataset folder containing scRNA_data.h5ad, spatial_data.h5ad, train_config.yaml')
    p.add_argument('--study-name', default='tangram_optuna_study', help='Optuna study name')
    p.add_argument('--storage', default=None, help='Optuna storage URL (e.g. sqlite:///path/to.db). If omitted, sqlite DB is created under output folder')
    p.add_argument('--n-trials', type=int, default=40)
    p.add_argument('--timeout', type=float, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--sampler', choices=['tpe', 'random'], default='tpe')
    p.add_argument('--dry-run', action='store_true', help='Only check files and config, do not run tuning')
    args = p.parse_args(argv)

    dataset_folder = os.path.abspath(args.dataset_folder)
    if not os.path.isdir(dataset_folder):
        print(f"Dataset folder not found: {dataset_folder}")
        sys.exit(1)

    # validate expected files
    sc_path, st_path, cfg_path = validate_dataset_folder(dataset_folder)

    dataset_name = os.path.basename(dataset_folder.rstrip(os.sep))
    print(f"Dataset: {dataset_name}")
    print(f"Single-cell file: {sc_path}")
    print(f"Spatial file: {st_path}")
    print(f"Config file: {cfg_path}")

    # load config
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # hardcode config
    config['lambda_sparsity_g1'] = 0
    config['num_epochs'] = 40  # 200

    random_state = config.get('random_state', None)

    # quick split genes to get train/val lists
    train_genes, val_genes = tgl.split_train_val_genes(sc.read_h5ad(sc_path), sc.read_h5ad(st_path), random_state=random_state)

    # infer output root and create output dir
    output_root = infer_output_root(dataset_folder)
    output_dir = os.path.join(output_root, 'optuna_tests', dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # prepare storage
    if args.storage is None:
        study_db = os.path.join(output_dir, f"{args.study_name}.db")
        storage = f"sqlite:///{study_db}"
    else:
        storage = args.storage

    print(f"Outputs will be written to: {output_dir}")
    print(f"Optuna storage: {storage}")

    if args.dry_run:
        print("Dry run: validation complete. Exiting.")
        return

    # load AnnData objects for tuning (read once)
    adata_sc = sc.read_h5ad(sc_path)
    adata_st = sc.read_h5ad(st_path)

    # call tuning
    best_value, best_params, trials_df = tgl.tune_loss_coefficients(
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

    # save results
    out_best = os.path.join(output_dir, f"{args.study_name}_best_params.yaml")
    out_trials = os.path.join(output_dir, f"{args.study_name}_trials.csv")
    with open(out_best, 'w') as f:
        yaml.dump({'best_value': best_value, 'best_params': best_params}, f)
    trials_df.to_csv(out_trials, index=False)

    print(f"Saved best params to: {out_best}")
    print(f"Saved trials dataframe to: {out_trials}")


if __name__ == '__main__':
    main()
