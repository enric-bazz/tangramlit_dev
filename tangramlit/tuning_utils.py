"""
Hyperparameter tuning functions for Tangram using the Lightning framework.
"""
import optuna
import lightning.pytorch as lp
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from tangramlit.mapping_datamodule import *
from tangramlit.mapping_lightningmodule import *
from tangramlit.mapping_utils import validate_mapping_inputs
from tangramlit.mapping_lightningmodule import MapperLightning
from tangramlit.mapping_datamodule import MyDataModule

def tune_loss_coefficients(
        adata_sc,
        adata_st,
        input_genes=None,
        train_genes_names=None,
        val_genes_names=None,
        cluster_label=None,
        learning_rate=0.1,
        num_epochs=100,
        random_state=None,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_l1=0,
        lambda_l2=0,
        lambda_sparsity_g1=0,
        lambda_neighborhood_g1=0,
        lambda_getis_ord=0,
        lambda_moran=0,
        lambda_geary=0,
        lambda_ct_islands=0,
        filter=False,  # filter related terms deactivated
        target_count=None,
        lambda_count=0,
        lambda_f_reg=0,
        study_name="loss_coefficients_tuning",
        storage=None,
        n_trials=30,
        timeout=None,
        resume=False,
        sampler_type='tpe',  # 'random' or 'tpe'
        pruner_type='median',  # 'median' or 'successive_halving'
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
        filter=False,  # fix filter related terms to deactivate it
        lambda_count=0,
        lambda_f_reg=0,
        target_count=None,
        )
  
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
    if sampler_type == 'random':
        sampler = optuna.samplers.RandomSampler(seed=random_state)
    elif sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=5,
            # seed=random_state,
            multivariate=True,  # joint sampling of correlated params
            group=True,  # groups correlated parameters
            # constant_liar=True,  # for parallel computing
        )
    if pruner_type == 'median':
        pruner = optuna.pruners.MedianPruner()
    elif pruner_type == 'successive_halving':
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=2,
            reduction_factor=3,
        )
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

    trials_df = study.trials_dataframe()

    return study.best_params, trials_df

    
    


def objective(
    trial: optuna.trial.Trial, 
    datamodule: MyDataModule, 
    training_config: dict = {},
):

    # Binary on/off switches
    # use_g1 = trial.suggest_categorical("use_g1", [0, 1])
    # use_sparsity_g1 = trial.suggest_categorical("use_sparsity_g1", [0, 1])

    # # Apply binary switches
    # training_config['lambda_g1'] = training_config['lambda_g1'] * use_g1
    # training_config['lambda_sparsity_g1'] = training_config['lambda_sparsity_g1'] * use_sparsity_g1

    # Lambda coefficient sampling
    lambda_max = 1

    # lambda_d
    if training_config.get('lambda_d', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_d', 1e-6, 1e-3, log=True)
        training_config['lambda_d'] = lambda_coeff_val

    # lambda_g2
    if training_config.get('lambda_g2', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_g2', 1e-2, lambda_max, log=True)
        training_config['lambda_g2'] = lambda_coeff_val

    # lambda_r
    if training_config.get('lambda_r', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_r', 1e-10, 1e-4, log=True)
        training_config['lambda_r'] = lambda_coeff_val

    # lambda_l1
    if training_config.get('lambda_l1', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_l1', 1e-18, 1e-12, log=True)
        training_config['lambda_l1'] = lambda_coeff_val

    # lambda_l2
    if training_config.get('lambda_l2', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_l2', 1e-21, 1e-12, log=True)
        training_config['lambda_l2'] = lambda_coeff_val

    # lambda_sparsity_g1
    # if training_config.get('lambda_sparsity_g1', 0) > 0:
    #     lambda_coeff_val = trial.suggest_float('lambda_sparsity_g1', 1e-2, 1e2, log=True)
    #     training_config['lambda_sparsity_g1'] = lambda_coeff_val

    # lambda_neighborhood_g1
    if training_config.get('lambda_neighborhood_g1', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_neighborhood_g1', 1e-2, 10, log=True)
        training_config['lambda_neighborhood_g1'] = lambda_coeff_val

    # lambda_getis_ord
    if training_config.get('lambda_getis_ord', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_getis_ord', 1e-2, 10, log=True)
        training_config['lambda_getis_ord'] = lambda_coeff_val

    # lambda_moran
    if training_config.get('lambda_moran', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_moran', 1e-2, 10, log=True)
        training_config['lambda_moran'] = lambda_coeff_val

    # lambda_geary
    if training_config.get('lambda_geary', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_geary', 1e-2, 10, log=True)
        training_config['lambda_geary'] = lambda_coeff_val

    # lambda_ct_islands
    if training_config.get('lambda_ct_islands', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_ct_islands', 1e-2, lambda_max, log=True)
        training_config['lambda_ct_islands'] = lambda_coeff_val

    # Init mapper with lambdas as sampled by optuna
    model = MapperLightning(**training_config)

    # Create pruning callback if available
    pruning_cb = PyTorchLightningPruningCallback(trial=trial, monitor='val_score')  # prune on score

    # Initialize trainer
    trainer = lp.Trainer(
        callbacks=pruning_cb,
        max_epochs=training_config["num_epochs"],
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        check_val_every_n_epoch=4,  # validate every training epocsh for faster pruning
    )

    # Fit model (trainer will call datamodule.prepare_data/setup as needed too)
    trainer.fit(model, datamodule=datamodule)

    # After training, retrieve the monitored metric from logged metrics
    metrics = trainer.callback_metrics
    val_score = metrics['val_score']
    # val_auc = metrics['val_AUC']
    
    # val = 1 * val_score + 1 * val_auc  # combined metric

    if val_score is None:
        # If metrics not found, raise to let Optuna mark trial as failed
        raise RuntimeError(f"Monitored metrics 'val_score' not found among trainer.callback_metrics: {list(metrics.keys())}")

    return val_score

# def tune_loss_coefficients_with_filter()
#     # This function can be implemented similarly to tune_loss_coefficients,
#     # but with the filter-related terms activated and tuned.
#     pass

# def objective_filter(trial: optuna.trial.Trial,
#                      datamodule: MyDataModule,
#                      training_config: dict = {},
#                      ):
#     # Similar to the objective function above, but includes tuning for filter-related terms.
#     if not training_config["filter"]:
#         raise ValueError("Filter must be enabled for this objective function.")
    
#     # Set filter-related hyperparameters (target_count defaults to adata_st.n_obs)
#     training_config["filter"] = trial.suggest_categorical("filter", [False, True])  # binary switch
#     training_config["lambda_count"] = trial.suggest_float("lambda_count", 1e-5, 1e-3, log=True)
#     training_config["lambda_f_reg"] = trial.suggest_float("lambda_f_reg", 1e-3, 1e-1, log=True)