"""
Mapping functions for Tangram using the Lightning framework.
"""
import logging
import numpy as np
import scanpy as sc
import torch
import warnings
from anndata import AnnData
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


from ..data.paired_data_module import PairedDataModule
from .lit_mapper import MapperLightning, EpochProgressBar


def validate_mapping_inputs(
        adata_sc,
        adata_st,
        input_genes,
        train_genes_names,
        val_genes_names,
        filter,
        learning_rate=0.1,
        num_epochs=1000,
        random_state=None,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_l1=0,
        lambda_l2=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        lambda_sparsity_g1=0,
        lambda_neighborhood_g1=0,
        lambda_getis_ord=0,
        lambda_moran=0,
        lambda_geary=0,
        lambda_ct_islands=0,
        cluster_label=None,
):
    """
    Validates inputs for cell-to-space mapping functions in the Tangram framework.

    Args:
        Same as map_cells_to_space()

    Returns:
        hyperparameters (dict): Dictionary of hyperparameters for the LightningModule training.
        Includes: filter, num_epochs, learning_rate, random_state, all lambdas, targe_count, cluster_label.
        Weights for refinement terms are added in map_cells_to_space().
        The anndata objects and gene indexes are only validated (not returned in the output dict).

    Raises:
        ValueError: If inputs are invalid or incompatible
    """

    # Check invalid values for arguments
    if lambda_g1 + lambda_sparsity_g1 == 0:
        raise ValueError("Either one of lambda_g1 or lambda_sparsity_g1 must be > 0.")

    # Validate anndata objects
    if not isinstance(adata_sc, AnnData) or not isinstance(adata_st, AnnData):
        raise ValueError("Both adata_sc and adata_st must be AnnData objects.")

    # Extract common genes names
    sc_genes = set(adata_sc.var_names.str.lower())
    st_genes = set(adata_st.var_names.str.lower())
    common_genes = sc_genes.intersection(st_genes)
    
    # Check that input_genes is a subset of overlapping genes
    if input_genes is not None:
        if not set(input_genes).issubset(common_genes):
            raise ValueError("input_genes must be a subset of the common genes set.")

    # Check that train_genes_names and val_genes_names are valid and do not intersect
    if train_genes_names is not None and val_genes_names is not None:
        if not (set(train_genes_names) <= common_genes and set(val_genes_names) <= common_genes):
            raise ValueError("train_genes_names and val_genes_names must be valid indices.")
        elif set(train_genes_names).intersection(set(val_genes_names)):
            raise ValueError("train_genes_names and val_genes_names must not intersect.")

    # Validate training numerical parameters
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not (random_state is None
            or isinstance(random_state, (int, np.integer, np.random.RandomState, np.random.Generator, torch.Generator))
    ):
        raise ValueError("random_state must be an int, None, or a random number generator")

    # CT islands enforcement
    if lambda_ct_islands > 0:
        if cluster_label is None or cluster_label not in adata_sc.obs.keys():
            raise ValueError(
                "cluster_label must be specified and in `adata_sc.obs.keys()` to use the cell type islands loss term."
            )
    # Check for missing values in cluster_label
    if cluster_label is not None :
        n_invalid = adata_sc.obs[cluster_label].isna().sum()
        if n_invalid > 0:        
            # Get boolean mask of valid cells
            valid_cells = ~adata_sc.obs[cluster_label].isna()
                    
            logging.warning(f"Found {n_invalid} cells in adata_sc with NaN `cluster_label` annotations. These cells will be removed.")

            # Remove cellss with NaN labels
            adata_sc._inplace_subset_obs(valid_cells)

    # Check spatial coordinates
    if 'spatial' not in adata_st.obsm.keys():
        raise ValueError("'spatial' key is missing in `adata_st.obsm`.")
        
    # Check for NaN values in spatial coordinates (squidpy graph does not work otherwise)
    if np.any(np.isnan(adata_st.obsm['spatial'])):
        # Get indices of rows without NaN values
        valid_spots = ~np.any(np.isnan(adata_st.obsm['spatial']), axis=1)
        n_invalid = np.sum(~valid_spots)
        
        logging.warning(f"Found {n_invalid} spots in adata_st with NaN coordinates. These spots will be removed.")
        
        # Remove spots with NaN coordinates
        adata_st._inplace_subset_obs(valid_spots)

    # Filter inputs
    if filter:
        if not all([lambda_f_reg, lambda_count]):
            raise ValueError(
                "lambda_f_reg and lambda_count must be specified if cell filter is active."
            )
        if not target_count:
            target_count = adata_st.shape[0]  # after spot removal
            logging.info(f"target_count missing from input is set to adata_st.shape[0] = {target_count}.")

    # Create hyperparameters dictionary for all next calls
    hyperparameters = {}
    hyperparameters['filter'] = filter  # filter
    hyperparameters['learning_rate'] = learning_rate  # learning rate
    hyperparameters['num_epochs'] = num_epochs  # number of epochs
    hyperparameters['random_state'] = random_state  # RNG seed

    # Loss
    loss_coeffs = {
        "lambda_d": lambda_d,  # KL (ie density) term
        "lambda_g1": lambda_g1,  # gene-voxel cos sim
        "lambda_g2": lambda_g2,  # voxel-gene cos sim
        "lambda_r": lambda_r,  # regularizer: penalize entropy
        "lambda_l1": lambda_l1,  # l1 regularizer
        "lambda_l2": lambda_l2,  # l2 regularizer
        "lambda_sparsity_g1": lambda_sparsity_g1,  # sparsity wighted term
        "lambda_neighborhood_g1": lambda_neighborhood_g1,  # neighborhood weighted term
        "lambda_getis_ord": lambda_getis_ord,  # Getis-Ord G* weighting
        "lambda_moran": lambda_moran,  # Moran's I weighting
        "lambda_geary": lambda_geary,  # Geary's C weighting
        "lambda_ct_islands": lambda_ct_islands,  # ct islands enforcement
    }
    hyperparameters.update(loss_coeffs)
    if filter:
        filter_terms = {
            "lambda_count": lambda_count,  # regularizer: enforce target number of cells
            "lambda_f_reg": lambda_f_reg,  # regularizer: push sigmoid values to 0,1
            "target_count": target_count,  # target number of cells
        }
        hyperparameters.update(filter_terms)

    return hyperparameters


def map_cells_to_space(
        adata_sc,
        adata_st,
        input_genes=None,
        train_genes_names=None,
        val_genes_names=None,
        filter=False,
        learning_rate=0.1,
        num_epochs=1000,
        random_state=None,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_l1=0,
        lambda_l2=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        lambda_sparsity_g1=0,
        lambda_neighborhood_g1=0,
        lambda_getis_ord=0,
        lambda_moran=0,
        lambda_geary=0,
        lambda_ct_islands=0,
        cluster_label=None,
        experiment_name="",
):
    """
    Map single cell data (`adata_sc`) on spatial data (`adata_st`).

    Args:
        adata_sc (AnnData): single cell data
        adata_st (AnnData): gene spatial data
        input_genes (list): Optional. Training gene list. Must be a subset of the genes shared between modalities.
        train_genes_names (list): Optional. Gene indices used for training from the training gene list. Must be a subset of the genes shared between modalities.
        val_genes_names (list): Optional. Gene indices used for validation from the training gene list. Must be a subset of single cell data genes, possibly not in the spatial data.
        filter (bool): Whether the cell filter is active or not. Default is False.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 0.
        lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
        lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. An higher entropy promotes probabilities of each cell peaked over a narrow portion of space. lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
        lambda_l1 (float): Optional. Strength of L1 regularizer. Default is 0.
        lambda_l2 (float): Optional. Strength of L2 regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when 'filter' is True.
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when 'filter' is True. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None (internally set to adata_st.shape[0]). Only valid when 'filter' is True'.
        lambda_sparsity_g1 (float): Optional. Strength of sparsity weighted gene expression comparison. Default is 0.
        lambda_neighborhood_g1 (float): Optional. Strength of neighborhood weighted gene expression comparison. Default is 0.
        lambda_getis_ord (float): Optional. Strength of Getis-Ord G* preservation. Default is 0.
        lambda_geary (float): Optional. Strength of Geary's C preservation. Default is 0.
        lambda_moran (float): Optional. Strength of Moran's I preservation. Default is 0.
        lambda_ct_islands (float): Optional. Strength of ct islands enforcement. Default is 0. 
        cluster_label (str): Optional. Field in `adata_sc.obs` used for aggregating single cell data. Only valid for lambda_ct-islands > 0. Default is None.

    Returns:
        A cell-by-spot AnnData containing the probability of mapping cell i on spot j.
        The `uns` field of the returned AnnData contains the training genes.
        The loss terms history is stored in adata_map.uns['training_history'].
        If 'filter' is True, the `uns` field also contains the filter values and the number of selected cells during each epoch.
        The final filter values are stored in adata_map.obs['F_out'].
        The final LightningModule and LightningDataModule objects are also returned.
    """

    # Input control function
    hyperparameters = validate_mapping_inputs(**{k: v for k, v in locals().items() if k in validate_mapping_inputs.__code__.co_varnames})  # pass all args

    # Initialize the model
    model = MapperLightning(**hyperparameters)

    # Initialize DataModule
    data = PairedDataModule(adata_sc=adata_sc,
                        adata_st=adata_st,
                        input_genes=input_genes,
                        train_genes_names=train_genes_names,
                        val_genes_names=val_genes_names,
                        cluster_label=cluster_label,
                        )

    # Customize ModelCheckpoint callback to avoid memory blow-up
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{experiment_name}/",
        filename='{epoch}-{val_score:.3f}',
        monitor="val_score",
        verbose=True,
        save_last=False,  # save last checkpoint separated from top k
        save_top_k=0,  # top k models out of all checkpoints
        mode="max",  # update with higher scores
        auto_insert_metric_name=True,
        save_weights_only=True,  # set = False to store optimizer, scheduler states
        every_n_epochs=50,
        save_on_train_epoch_end=False,  # val_score is in validation_step()
    )

    # Set early stopper
    early_stop = EarlyStopping(
        monitor="val_score",  # monitor validation score
        min_delta=0.001,  # score minimum improvement loss, 0.001
        patience=20,  # related to check_val_every_n_epoch, 20
        verbose=False,
        mode="max",
        check_on_train_epoch_end=False,  # val_score is in validation_step()
    )

    # Set lr monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create TB logger for training
    train_logger = TensorBoardLogger(save_dir="tb_logs", name=f"train_{experiment_name}")

    # Initialize trainer with automatic device detection
    trainer = Trainer(
        logger=train_logger,
        callbacks=[EpochProgressBar(), lr_monitor, early_stop],
        min_epochs=200,  # 200
        max_epochs=num_epochs,  # num_epochs
        log_every_n_steps=1,  # log every training step == epoch
        enable_checkpointing=False,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,  # validation loop after every N training epochs
        accelerator="auto",  # Automatically detect GPU/CPU (cuda, mps, or cpu)
        devices="auto",  # Automatically detect number of GPUs
        precision="32-true",  # Use single precision (32-bit float) for stability, change to "16-mixed" for mixed precision on GPU
    )

    # Train the model
    trainer.fit(model, datamodule=data)

    # Get the final mapping matrix
    with torch.no_grad():
        if model.hparams.filter:
            _, mapping_filt, filter_probs = model()  # Unpack values (skip unfiltered M matrix)
            final_mapping = mapping_filt.cpu().numpy()  # Turn into numpy array
            final_filter = filter_probs.cpu().numpy()  # Turn into numpy array
        else:
            final_mapping = model().cpu().numpy()


    # Save results
    logging.info("Saving results...")
    # Retrieve final training genes (after preprocessing) from DataModule
    training_genes = data.adata_sc.uns['training_genes']  # available after trainer.fit()
    # Create AnnData object with the final mapping matrix
    adata_map = sc.AnnData(
        X=final_mapping,
        obs=data.adata_sc[:, training_genes].obs.copy(),
        var=data.adata_st[:, training_genes].obs.copy(),
    )
    # Store training genes in adata_map.uns['training_genes']
    adata_map.uns['training_genes'] = training_genes
    # Store final filter values in adata_map.obs['F_out']
    if filter:
        adata_map.obs["F_out"] = final_filter
    
    # Create training history dictionary
    training_history = {
        'total_loss': model.loss_history['loss'],
        'main_loss': model.loss_history['main_loss'],
        'vg_reg': model.loss_history['vg_reg'],
        'kl_reg': model.loss_history['kl_reg'],
        'entropy_reg': model.loss_history['entropy_reg'],
        'l1_term': model.loss_history['l1_term'],
        'l2_term': model.loss_history['l2_term'],
        'sparsity_term': model.loss_history['sparsity_term'],
        'neighborhood_term': model.loss_history['neighborhood_term'],
        'getis_ord_term': model.loss_history['getis_ord_term'],
        'moran_term': model.loss_history['moran_term'],
        'geary_term': model.loss_history['geary_term'],
        'ct_island_term': model.loss_history['ct_island_term'],
    }
    # Add filter terms only if filter was used
    if model.hparams.filter:
        filter_terms = {
            'count_reg': model.loss_history['count_reg'],
            'filt_reg': model.loss_history['filt_reg']
        }
        training_history.update(filter_terms)
        # Include filter training history logs
        adata_map.uns['filter_history'] = {
            'filter_values': model.filter_history['filter_values'],
            'n_cells': model.filter_history['n_cells']
        }
    # Store training history in adata_map.uns['training_history']
    adata_map.uns['training_history'] = training_history

    # Store validation history in adata_map.uns['validation_history']
    adata_map.uns['validation_history'] = model.val_history

    return adata_map, model, data


def run_multiple_mappings(
    adata_sc,
    adata_st,
    config,
    n_runs=10,
    compute_mapping_cube=True,
    compute_filtered_cube=False,
    compute_filter_square=False,
    ):
    """
    Runs multiple mappings using the same configuration and returns the final alignments cube.
    If in filter mode the final filters matrix is also returned.
    Args are added to control what tensors are stored for each run: storing n_runs mapping matrices might be memory-intensive.

    Args:
        adata_sc (AnnData): single cell data
        adata_st (AnnData): spatial data
        config (dict): Dictionary containing configuration parameters for the mapping.
            Object must match keys in map_cell_sto_space() hyperparameters dictionary.
        n_runs (int): Number of runs to perform. Default is 10.
        compute_mapping_cube (bool): Whether to compute the final mapping cube. Default is True.
        compute_filtered_cube (bool): Whether to compute the final filtered cube. Default is False.
        compute_filter_square (bool): Whether to compute the final filter square. Default is False.

    Returns:
        Specified arrays stacked into a 2nd or 3rd dimension.
    """
    # Checks
    if (compute_filtered_cube or compute_filter_square) and config['mode'] != 'filter':
        raise ValueError("compute_filtered_cube and compute_filter_square can only be used in filter mode.")

    # Call the data module to retrieve batch size
    datamodule = PairedDataModule(adata_sc=adata_sc,
                        adata_st=adata_st,
                        #input_genes=input_genes,
                        #refined_mode=mode == "refined",
                        #train_genes_names=train_genes_names,
                        #val_genes_names=val_genes_names,
                        )

    # Last arg is mandatory otherwise the validation_step() method is called with no val_dataset
    print("\nEntering multiple runs training:")

    # Store dimensions
    n_cells = adata_sc.n_obs
    n_spots = adata_st.n_obs

    # Init mapping cube to final size (accessible) to pre-allocate memory
    if compute_mapping_cube:
        print(f"Allocating mapping cube of size = {n_cells * n_spots * n_runs * 8 / (1024 ** 2):.2f} MiB ...")
        mapping_matrices_cube = np.zeros((n_cells, n_spots, n_runs))
        print(f"Done")

    # Init filtered mapping cube to final size (accessible) to pre-allocate memory
    if compute_filtered_cube:
        print(f"Allocating filtered mapping cube of size = {n_cells * n_spots * n_runs * 8 / (1024 ** 2):.2f} MiB ...")
        filtered_matrices_cube = np.zeros((n_cells, n_spots, n_runs))
        print(f"Done")

    # Init filters matrix to final size
    if compute_filter_square:
        print(f"Allocating filters square of size =  {n_cells * n_runs * 8 / (1024**2):.2f} MiB ...")
        filters_square = np.zeros((datamodule.adata_sc.n_obs, n_runs))
        print(f"Done")

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}...")
        # Set run RNG seed (optional)
        config['random_state'] = run
        #config['random_state'] = config['random_state'] + run if 'random_state' in config else [run,]

        # Initialize trainer here (otherwise after 1 run max_epochs is reached already)
        trainer = Trainer(
            max_epochs=config['num_epochs'],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_val_batches=0,  # disables validation during fit
            accelerator="auto",  # Automatically detect GPU/CPU
            devices="auto",  # Automatically detect number of devices
            precision="32-true",  # Single precision for stability
        )

        # Initialize the model
        model = MapperLightning(**config)

        # Train the model
        trainer.fit(model, datamodule=datamodule)

        # Get the final mapping matrix
        with torch.no_grad():
            if model.hparams.filter:
                mapping, filtered_mapping, filter_probs = model()  # Unpack values (skip filtered M matrix)
                final_mapping = mapping.cpu().numpy()
                final_filtered_mapping = filtered_mapping.cpu().numpy()
                final_filter = filter_probs.cpu().numpy()
            else:
                final_mapping = model().cpu().numpy()

        # Store mapping to cube
        if compute_mapping_cube:
            mapping_matrices_cube[:,:,run] = final_mapping
        if compute_filtered_cube:
            filtered_matrices_cube[:,:,run] = final_filtered_mapping

        # Store filter to square
        if compute_filter_square:
            filters_square[:,run] = final_filter

    # Prepare results
    result = []
    if compute_mapping_cube:
        result.append(mapping_matrices_cube)
    if compute_filtered_cube:
        result.append(filtered_matrices_cube)
    if compute_filter_square:
        result.append(filters_square)

    return result