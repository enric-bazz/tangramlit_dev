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

import sys
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

from tangramlit.hpo.lambda_optuna_study import tune_loss_coefficients


def main():
    # Set study parameters
    data_path = "C:/Users/enric/desktop/tangram_repo_dump/"
    adata_sc = ad.read_h5ad(data_path + "myDataCropped/test_sc_crop.h5ad")
    adata_st = ad.read_h5ad(data_path + "myDataCropped/slice200_norm_reduced.h5ad")
    study_name = "tangram_mapping_tuning_prova_prova_prova"
    study_path = "C:/Users/enric/desktop/tangram_repo_dump/"
    storage = f"sqlite:///{study_path}tangram_mapping_tuning.db"
    n_trials = 1    
    resume = True

    with open("C:/Users/enric/desktop/tangram_repo_dump/MyDataCropped/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    random_state = config["random_state"]

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Get train/val genes (0.8 split)
    train_genes, val_genes, _ = tgl.split_train_val_test_genes(adata_sc, adata_st, random_state=random_state)

    best_value, best_params, trials_df = tune_loss_coefficients(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=None,
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        **config,
        study_name=study_name,
        storage=storage,
        n_trials=n_trials,
        resume=True,
        )

    with open(f"{study_path}{study_name}_best_params.yaml", "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)

    trials_df.to_csv(f"{study_path}{study_name}_trials.csv", index=False, sep=';')
 


if __name__ == "__main__":
    main()