"""
Test script for Optuna tuning public API.
"""

import sys
import logging
import yaml
import optuna
import anndata as ad
import tangramlit as tgl


def main():
    # Set study parameters
    data_path = "C:/Users/enric/tangramlit_dev/data/"
    adata_sc = ad.read_h5ad(data_path + "test_sc.h5ad")
    adata_st = ad.read_h5ad(data_path + "test_slice200.h5ad")
    study_name = "api_test_study"
    study_path = "C:/Users/enric/tangramlit_dev/out/"
    storage = f"sqlite:///{study_path}tangram_tuning_api_test.db"
    n_trials = 2    
    resume = False

    with open(data_path + "train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    random_state = config["random_state"]

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Get train/val genes (0.8 split)
    train_genes, val_genes, _ = tgl.split_train_val_test_genes(adata_sc, adata_st, random_state=random_state)

    best_value, best_params, trials_df = tgl.tune_loss_coefficients(
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