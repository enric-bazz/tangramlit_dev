"""Optuna tuning script for the Tangram `MapperLightning` LightningModule.

Usage of the CLI main (examples):
  - Run 50 trials, using 1 GPU (if available):
      python optuna_tune.py --adata-sc path/to/sc.h5ad --adata-st path/to/st.h5ad --n-trials 50 --gpus 1

  - Resume a study saved to `sqlite:///optuna_study.db`:
      python optuna_tune.py --adata-sc sc.h5ad --adata-st st.h5ad --study-name my_study --storage sqlite:///optuna_study.db --resume

This script samples a concise search space (regularizer weights),
creates a `MapperLightning` with sampled hyperparameters, trains it with a PyTorch Lightning
Trainer and uses Optuna's pruning callback to stop unpromising trials early.
"""

import argparse
import os
import sys
from typing import Optional
import logging

import optuna
import torch
import lightning.pytorch as lp

import optuna
import tangramlit as tgl
import scanpy as sc
import anndata as ad
import yaml
import numpy as np
import pandas as pd



def main():
    # Set study parameters
    data_path = "C:/Users/enric/tangram/"
    adata_sc = ad.read_h5ad(data_path + "myDataCropped/test_sc_crop.h5ad")
    adata_st = ad.read_h5ad(data_path + "myDataCropped/slice200_norm_reduced.h5ad")
    study_name = "tangram_mapping_tuning_6"
    study_path = "F:/optuna_tests/"
    storage = f"sqlite:///{study_path}tangram_mapping_tuning.db"
    n_trials = 1
    resume = True

    with open("C:/Users/enric/tangramlit_dev/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    random_state = config["random_state"]

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Get train/val genes (0.8 split)
    train_genes, val_genes = tgl.split_train_val_genes(adata_sc, adata_st, random_state=random_state)

    best_params, trials_df = tgl.tune_mapping_loss_coefficients(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=None,
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        **config,
        study_name=study_name,
        storage=storage,
        n_trials=n_trials,
        direction="maximize",
        monitor="val_score",
        timeout=None,
        resume=resume,
        )

    with open(f"{study_path}{study_name}_best_params.yaml", "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)

    trials_df.to_csv(f"{study_path}{study_name}_trials.csv", index=False, sep=';')  # or ,




def main_for_cli(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Tune MapperLightning hyperparameters with Optuna")
    parser.add_argument("--adata-sc", required=True, help="Path to single-cell AnnData (.h5ad)")
    parser.add_argument("--adata-st", required=True, help="Path to spatial AnnData (.h5ad)")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="Study timeout in seconds")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (if available)")
    parser.add_argument("--max-epochs", type=int, default=200, help="Number of epochs per trial")
    parser.add_argument("--study-name", type=str, default="tangram_optuna_study")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db", help="Optuna storage URL to persist study")
    parser.add_argument("--resume", action="store_true", help="Resume existing study if present in storage")
    parser.add_argument("--monitor", type=str, default="val_score", help="Metric to monitor from validation (default: val_score)")
    parser.add_argument("--direction", type=str, choices=["minimize", "maximize"], default="maximize")
    args = parser.parse_args(argv)

    # Load AnnData objects
    adata_sc = sc.read_h5ad(args.adata_sc)
    adata_st = sc.read_h5ad(args.adata_st)

    study_name = args.study_name
    storage = args.storage      
    n_trials = args.n_trials
    resume = args.resume
    timeout = args.timeout
    direction = args.direction
    max_epochs = args.max_epochs
    monitor = args.monitor
    data_path = os.path.dirname(args.adata_sc) + "/"
    study_path = os.path.dirname(storage.replace("sqlite:///", "")) + "/"

    with open(f"{data_path}train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    random_state = config["random_state"]

    # Get train/val genes (0.8 split)
    train_genes, val_genes = tgl.get_train_val_genes(adata_sc, adata_st, random_state=random_state)

    best_params = tgl.tune_mapping_loss_coefficients(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=None,
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        **config,
        study_name=study_name,
        storage=storage,
        n_trials=n_trials,
        direction="maximize",
        monitor="val_score",
        timeout=None,
        resume=resume,
        )

    with open(f"{study_path}best_params.yaml", "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)

 


if __name__ == "__main__":
    main()