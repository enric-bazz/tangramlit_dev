"""
Hyperparameter tuning functions for Tangram using the Lightning framework.
"""
import optuna
import torch
import lightning.pytorch as lp
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
import gc

from tangramlit.data import PairedDataModule
from tangramlit.mapping import MapperLightning
from tangramlit.mapping.trainer import validate_mapping_inputs

def tune_loss_coefficients(
        adata_sc,                                       ######### tangram args
        adata_st,
        input_genes=None,
        train_genes_names=None,
        val_genes_names=None,
        cluster_label=None,
        learning_rate=0.1,
        num_epochs=100,
        random_state=None,
        lambda_d=1,
        lambda_g1=1,
        lambda_g2=1,
        lambda_r=1,
        lambda_l1=1,
        lambda_l2=1,
        lambda_sparsity_g1=0,
        lambda_neighborhood_g1=1,
        lambda_getis_ord=1,
        lambda_moran=1,
        lambda_geary=1,
        lambda_ct_islands=1,
        filter=True,
        target_count=None,
        lambda_count=1e-5,
        lambda_f_reg=1e-5,  
        study_name="loss_coefficients_tuning",        ########### optuna args
        storage='/nfsd/sysbiobig/bazzaccoen/tangramlit_dev/',
        n_trials=40,
        timeout=None,
        resume=False,
        sampler_type='tpe',  # 'random' or 'tpe'
):
    
    # Input control function
    hyperparameters = validate_mapping_inputs(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=input_genes,
        train_genes_names=train_genes_names,
        val_genes_names=val_genes_names,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        random_state=random_state,
        lambda_d=lambda_d,
        lambda_g1=lambda_g1,
        lambda_g2=lambda_g2,
        lambda_r=lambda_r,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        lambda_sparsity_g1=lambda_sparsity_g1,
        lambda_neighborhood_g1=lambda_neighborhood_g1,
        lambda_getis_ord=lambda_getis_ord,
        lambda_moran=lambda_moran,
        lambda_geary=lambda_geary,
        lambda_ct_islands=lambda_ct_islands,
        cluster_label=cluster_label,
        filter=filter,
        lambda_count=lambda_count,
        lambda_f_reg=lambda_f_reg,
        target_count=target_count,
        )
  
    # Create datamodule
    datamodule = PairedDataModule(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=input_genes,
        train_genes_names=train_genes_names,
        val_genes_names=val_genes_names,
        cluster_label=cluster_label,
        )
    
    # Set up Optuna sampler and pruner
    if sampler_type == 'random':
        sampler = optuna.samplers.RandomSampler(seed=random_state)
    elif sampler_type == 'tpe':
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=5,
            seed=random_state,
            multivariate=True,  # joint sampling of correlated params
            group=True,  # groups correlated parameters
            constant_liar=True,  # for parallel computing
        )
    else:
        raise ValueError("Invalid sampler type. Valid typer are 'random' and 'tpe'")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=40, interval_steps=10)  # 5, 40, 10

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
        study.optimize(lambda t: objective(trial=t, datamodule=datamodule, training_config=hyperparameters),
                       n_trials=n_trials,
                       timeout=timeout)
    except KeyboardInterrupt:
        print("Optimization interrupted by user. Saving study.")

    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # clean memory
    del datamodule
    gc.collect()

    return study.best_value, study.best_params, study.trials_dataframe()

    
    


def objective(
    trial: optuna.trial.Trial, 
    datamodule: PairedDataModule, 
    training_config: dict = {},
):

    seed = trial.number + training_config['random_state']
    lp.seed_everything(seed, workers=True)

    # Lambda coefficient sampling

    # lambda_d
    if training_config.get('lambda_d', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_d', 1e-2, 10.0, log=True)
        training_config['lambda_d'] = lambda_coeff_val

    # lambda_g2
    if training_config.get('lambda_g2', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_g2', 0, 1.0, log=False)
        training_config['lambda_g2'] = lambda_coeff_val

    # lambda_r
    if training_config.get('lambda_r', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_r', 1e-12, 1e-4, log=True)
        training_config['lambda_r'] = lambda_coeff_val

    # lambda_l1
    if training_config.get('lambda_l1', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_l1', 1e-18, 1e-7, log=True)
        training_config['lambda_l1'] = lambda_coeff_val

    # lambda_l2
    if training_config.get('lambda_l2', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_l2', 1e-21, 1e-7, log=True)
        training_config['lambda_l2'] = lambda_coeff_val

    # lambda_neighborhood_g1
    if training_config.get('lambda_neighborhood_g1', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_neighborhood_g1', 0, 1.0, log=False)
        training_config['lambda_neighborhood_g1'] = lambda_coeff_val

    # lambda_getis_ord
    if training_config.get('lambda_getis_ord', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_getis_ord', 0, 1.0, log=False)
        training_config['lambda_getis_ord'] = lambda_coeff_val

    # lambda_moran
    if training_config.get('lambda_moran', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_moran', 0, 1.0, log=False)
        training_config['lambda_moran'] = lambda_coeff_val

    # lambda_geary
    if training_config.get('lambda_geary', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_geary', 0, 1.0, log=False)
        training_config['lambda_geary'] = lambda_coeff_val

    # lambda_ct_islands
    if training_config.get('lambda_ct_islands', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_ct_islands', 1e-3, 1e3, log=True)
        training_config['lambda_ct_islands'] = lambda_coeff_val

    # Init mapper with lambdas as sampled by optuna
    model = MapperLightning(**training_config)

    # Create pruning callback if available
    pruning_cb = PyTorchLightningPruningCallback(trial=trial, monitor='val_score')  # prune on score

    # Initialize trainer with automatic device detection
    trainer = lp.Trainer(
        callbacks=pruning_cb,
        max_epochs=training_config["num_epochs"],
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        check_val_every_n_epoch=4,
        accelerator="auto",  # Automatically detect GPU/CPU
        devices="auto" if not torch.cuda.is_available() else [0],  # Use single GPU per trial if available
        precision="32-true",  # Single precision for stability
    )

    # Fit model (trainer will call datamodule.prepare_data/setup as needed too)
    trainer.fit(model, datamodule=datamodule)

    # After training, retrieve the monitored metric from logged metrics
    metrics = trainer.callback_metrics
    val_score = metrics['val_score']

    if val_score is None:
        # If metrics not found, raise to let Optuna mark trial as failed
        raise RuntimeError(f"Monitored metrics 'val_score' not found among trainer.callback_metrics: {list(metrics.keys())}")

    # clean memory to avoid in-study build-up
    del pruning_cb
    del trainer
    del model
    gc.collect()

    return val_score