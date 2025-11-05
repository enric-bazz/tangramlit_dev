"""
Hyperparameter tuning functions for Tangram using the Lightning framework.
"""
import optuna
import lightning.pytorch as pl
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from tangramlit.mapping_datamodule import *
from tangramlit.mapping_lightningmodule import *
from tangramlit.mapping_utils import validate_mapping_inputs
from tangramlit.mapping_lightningmodule import MapperLightning
from tangramlit.mapping_datamodule import MyDataModule

def tune_mapping_loss_coefficients(
        adata_sc,
        adata_st,
        input_genes=None,
        train_genes_names=None,
        val_genes_names=None,
        cluster_label=None,
        learning_rate=0.1,
        num_epochs=1000,
        random_state=None,
        filter=False,
        target_count=None,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_l1=0,
        lambda_l2=0,
        lambda_count=1,
        lambda_f_reg=1,
        lambda_sparsity_g1=0,
        lambda_neighborhood_g1=0,
        lambda_getis_ord=0,
        lambda_moran=0,
        lambda_geary=0,
        lambda_ct_islands=0,
        study_name="tangram_mapping_tuning",
        storage=None,
        n_trials=50,
        direction="maximize",
        monitor="val_score",
        timeout=None,
        resume=False,
):
    
     # Input control function
    hyperparameters = validate_mapping_inputs(adata_sc,
        adata_st,
        input_genes,
        train_genes_names,
        val_genes_names,
        filter,
        learning_rate,
        num_epochs,
        random_state,
        lambda_d,
        lambda_g1,
        lambda_g2,
        lambda_r,
        lambda_l1,
        lambda_l2,
        lambda_count,
        lambda_f_reg,
        target_count,
        lambda_sparsity_g1,
        lambda_neighborhood_g1,
        lambda_getis_ord,
        lambda_moran,
        lambda_geary,
        lambda_ct_islands,
        cluster_label,)
  
   
    # Create datamodule
    datamodule = MyDataModule(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=input_genes,
        train_genes_names=train_genes_names,
        val_genes_names=val_genes_names,
        cluster_label=cluster_label,
        )
    
    # Set up Optuna sampler and pruner
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    # Define or load optuna study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=resume,
    )

    # Set info logging
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Optimize
    try:
        study.optimize(lambda t: objective(t, datamodule, monitor=monitor, training_config=hyperparameters),
                       n_trials=n_trials,
                       timeout=timeout)
    except KeyboardInterrupt:
        print("Optimization interrupted by user. Saving study.")

    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    trials_df = study.trials_dataframe()

    return study.best_params, trials_df

    
    


def objective(trial: optuna.trial.Trial, 
                datamodule: MyDataModule, 
                monitor="val_score",
                training_config: dict = {},
                ):

    # Lambda coefficient sampling
    lambda_max = 1

    # lambda_d
    if training_config.get('lambda_d', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_d', 1e-6, lambda_max, log=True)
        training_config['lambda_d'] = lambda_coeff_val

    # lambda_g2
    if training_config.get('lambda_g2', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_g2', 1e-3, lambda_max, log=True)
        training_config['lambda_g2'] = lambda_coeff_val

    # lambda_r
    if training_config.get('lambda_r', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_r', 1e-12, lambda_max, log=True)
        training_config['lambda_r'] = lambda_coeff_val

    # lambda_l1
    if training_config.get('lambda_l1', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_l1', 1e-18, lambda_max, log=True)
        training_config['lambda_l1'] = lambda_coeff_val

    # lambda_l2
    if training_config.get('lambda_l2', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_l2', 1e-21, lambda_max, log=True)
        training_config['lambda_l2'] = lambda_coeff_val

    # lambda_sparsity_g1
    if training_config.get('lambda_sparsity_g1', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_sparsity_g1', 1e-2, 1e2, log=True)
        training_config['lambda_sparsity_g1'] = lambda_coeff_val

    # lambda_neighborhood_g1
    if training_config.get('lambda_neighborhood_g1', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_neighborhood_g1', 1e-3, 10, log=True)
        training_config['lambda_neighborhood_g1'] = lambda_coeff_val

    # lambda_getis_ord
    if training_config.get('lambda_getis_ord', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_getis_ord', 1e-3, 10, log=True)
        training_config['lambda_getis_ord'] = lambda_coeff_val

    # lambda_moran
    if training_config.get('lambda_moran', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_moran', 1e-3, 10, log=True)
        training_config['lambda_moran'] = lambda_coeff_val

    # lambda_geary
    if training_config.get('lambda_geary', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_geary', 1e-3, 10, log=True)
        training_config['lambda_geary'] = lambda_coeff_val

    # lambda_ct_islands
    if training_config.get('lambda_ct_islands', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_ct_islands', 1e-3, lambda_max, log=True)
        training_config['lambda_ct_islands'] = lambda_coeff_val

    # If filter mode enabled, sample target_count and count regularizer
    if training_config["filter"]:
        training_config["filter"] = trial.suggest_categorical("filter", [False, True])
        training_config["lambda_count"] = trial.suggest_float("lambda_count", 1e-4, lambda_max, log=True)
        training_config["lambda_f_reg"] = trial.suggest_float("lambda_f_reg", 1e-3, lambda_max, log=True)


    # Init mapper with lambdas as sampled by optuna
    model = MapperLightning(**training_config)

    # Create pruning callback if available
    pruning_cb = PyTorchLightningPruningCallback(trial, monitor[0])  # prune on score

    # Initialize trainer
    trainer = pl.Trainer(
        callbacks=pruning_cb,
        max_epochs=training_config["num_epochs"],
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=False,
        check_val_every_n_epoch=1,  # validate every training epoch
    )

    # Fit model (trainer will call datamodule.prepare_data/setup as needed too)
    trainer.fit(model, datamodule=datamodule)

    # After training, retrieve the monitored metric from logged metrics
    metrics = trainer.callback_metrics
    val = {}
    # for term in monitor:
    if monitor in metrics.keys():
        val = metrics[monitor].item() if hasattr(metrics[monitor], "item") else float(metrics[monitor])
    if not val:
        # If metric not found, raise to let Optuna mark trial as failed
        raise RuntimeError(f"Monitored metric '{monitor}' not found among trainer.callback_metrics: {list(metrics.keys())}")

    return val