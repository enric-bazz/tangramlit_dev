"""
Mapping functions for Tangram using the Lightning framework.
"""
import pandas as pd
import sklearn
import warnings
from anndata import AnnData
import lightning.pytorch as lp
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.nn.functional import softmax, cosine_similarity


from . import utils as ut
import tangramlit.validation_metrics as vm
from .mapping_datamodule import *
from .mapping_lightningmodule import *


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
    data = MyDataModule(adata_sc=adata_sc,
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

    # Initialize trainer
    trainer = lp.Trainer(
        logger=train_logger,
        callbacks=[EpochProgressBar(), lr_monitor, early_stop],
        min_epochs=200,  # 200
        max_epochs=num_epochs,  # num_epochs
        log_every_n_steps=1,  # log every training step == epoch
        enable_checkpointing=False,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,  # validation loop after every N training epochs
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


def validate_mapping_experiment(model, datamodule, experiment_name=None):
    """
    Validate mapping experiment. The model must be already trained --> Run only after map_cells_to_space().

    Returns:
        validation_results (dict): Dictionary containing validation results with
        dict_keys(['val_AUC', 'val_SSIM', 'val_PCC', 'val_RMSE', 'val_JS']).
        Note that when called as 'results = validate_mapping_experiment(model, datamodule)', results is a list with
        element results[0] containing the dictionary.
    """
    val_logger = TensorBoardLogger("tb_logs", name=f"val_{experiment_name}")
    trainer = lp.Trainer(
        logger=val_logger,
        enable_progress_bar=False,  # disable progress bar
        num_sanity_val_steps=0,  # skip sanity check
    )
    validation_results = trainer.validate(model=model, datamodule=datamodule)

    return validation_results


def split_train_val_genes(adata_sc, adata_st, train_ratio=0.8, random_state=None):
    """
    Split genes into training and validation sets. Run before map_cells_to_space() and validate_mapping_experiment().

    Args:
        adata_sc (AnnData): Single-cell AnnData object.
        adata_st (AnnData): Spatial AnnData object.
        train_ratio (float): Proportion of genes to use for training. Default is 0.8.
        random_state (int): Random seed for reproducibility. Default is None.

    Returns:
        train_genes (list): List of training gene names.
        val_genes (list): List of validation gene names.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Set gene names to lowercase for case-insensitive matching
    sc_genes = {gene.lower(): gene for gene in adata_sc.var.index}
    st_genes = {gene.lower(): gene for gene in adata_st.var.index}

    # Get shared genes
    shared_genes = set(sc_genes.keys()) & set(st_genes.keys())

    # Split genes into training and validation sets
    train_genes = np.random.choice(list(shared_genes), size=int(len(shared_genes) * train_ratio), replace=False)
    val_genes = list(shared_genes - set(train_genes))

    return train_genes, val_genes


def project_sc_genes_onto_space(adata_map, datamodule):
    """
        Transfer gene expression from the single cell onto space.

        Args:
            adata_map (AnnData): cells-by-spots anndata object containing the mapping matrix.
            datamodule (LightningDataModule): LightningDataModule object containing the preprocessed single cell and spatial data.
                both are returned by map_cells_to_space().

        Returns:
            AnnData: spot-by-gene AnnData containing spatial gene expression from the single cell data.
    """
    # Subset single-cell genes to those marked as training or validation
    sc_var = datamodule.adata_sc.var
    # Prepare boolean Series for mask (handle missing columns)
    is_train = sc_var.get("is_training", pd.Series(False, index=sc_var.index))
    is_val = sc_var.get("is_validation", pd.Series(False, index=sc_var.index))
    # Ensure boolean dtype and fill NaNs
    is_train = is_train.fillna(False).astype(bool)
    is_val = is_val.fillna(False).astype(bool)

    gene_mask = (is_train | is_val).values
    if gene_mask.sum() == 0:
        raise ValueError("No genes selected for projection: no 'is_training' or 'is_validation' True in datamodule.adata_sc.var")

    # Create a local subset of the single-cell AnnData (do not mutate original datamodule)
    adata_sc_sub = datamodule.adata_sc[:, gene_mask].copy()

    # Make sc expression matrix dense for the subset
    if hasattr(adata_sc_sub.X, "toarray"):
        adata_sc_sub.X = adata_sc_sub.X.toarray()

    # Project sc expression matrix onto space (spots x genes_sub)
    X_space = adata_map.X.T @ adata_sc_sub.X

    # Create AnnData object with the projected spatial gene expression
    adata_ge = sc.AnnData(
        X=X_space,  # projected sc expression profiles
        obs=adata_map.var,  # spatial data spot IDs
        var=adata_sc_sub.var.copy(),  # selected sc gene metadata
        uns=adata_sc_sub.uns,  # unstructured fields of sc data 
    )

    # Annotate training and validation flags in adata_ge.var
    # If the original sc var had these columns, copy them; otherwise set False
    if "is_training" in adata_sc_sub.var.columns:
        adata_ge.var["is_training"] = adata_sc_sub.var["is_training"].astype(bool)
    else:
        adata_ge.var["is_training"] = False

    if "is_validation" in adata_sc_sub.var.columns:
        adata_ge.var["is_validation"] = adata_sc_sub.var["is_validation"].astype(bool)
    else:
        adata_ge.var["is_validation"] = False

    return adata_ge


def benchmark_mapping(adata_ge, datamodule):
    """Benchmark predicted spatial expression against true spatial expression.

    Compares generated spatial gene expression (`adata_ge`) with the true spatial
    expression contained in `datamodule.adata_st`. For each gene present in both
    datasets (and non-zero in both), the function computes:
      - cosine similarity (returned in column ``score``)
      - Pearson correlation coefficient (``PCC``)
      - Root mean squared error (``RMSE``)
      - Jensen-Shannon divergence (``JS``)
      - Structural Similarity Index (``SSIM``)

    The returned DataFrame also contains ``is_training``, ``is_validation``,
    ``sparsity_st``, ``sparsity_sc`` and ``sparsity_diff`` for downstream
    analysis.

    Notes:
      - Genes that are all-zero in either predicted or true matrices are excluded
        to avoid zero-norm problems when computing cosine similarity.

    Args:
        adata_ge (AnnData): generated spatial data returned by
            :func:`project_sc_genes_onto_space()` (spots x genes).
        datamodule (LightningDataModule): LightningDataModule containing the
            preprocessed single-cell and spatial data (used for ground-truth and
            sparsity annotations).

    Returns:
        pd.DataFrame: Per-gene metrics indexed by gene name. Columns include
        ``score``, ``PCC``, ``RMSE``, ``JS``, ``SSIM``, ``is_training``,
        ``is_validation``, ``sparsity_st``, ``sparsity_sc``, ``sparsity_diff``.
    """
    # Use genes present in the generated data (`adata_ge`) intersecting with spatial genes
    genes_ge = list(adata_ge.var_names)
    spatial_genes = set(datamodule.adata_st.var_names)
    common_genes = [g for g in genes_ge if g in spatial_genes]
    if len(common_genes) == 0:
        raise ValueError("No common genes between generated data (`adata_ge`) and spatial data to compare.")

    # Predicted spatial expression matrix for selected genes
    if hasattr(adata_ge.X, "toarray"):
        X_pred_full = adata_ge[:, common_genes].X.toarray()
    else:
        X_pred_full = adata_ge[:, common_genes].X
    # True spatial expression matrix for selected genes
    if hasattr(datamodule.adata_st.X, "toarray"):
        X_true_full = datamodule.adata_st[:, common_genes].X.toarray()
    else:
        X_true_full = datamodule.adata_st[:, common_genes].X

    # Exclude genes that are all-zero in either predicted or true matrices (avoid zero norms)
    nonzero_pred = ~np.all(X_pred_full == 0, axis=0)
    nonzero_true = ~np.all(X_true_full == 0, axis=0)
    valid_mask = nonzero_pred & nonzero_true
    if valid_mask.sum() == 0:
        raise ValueError("No genes with non-zero expression in both predicted and true spatial data to compare.")

    selected_idxs = np.where(valid_mask)[0]
    selected_genes = [common_genes[i] for i in selected_idxs]

    # Compute cosine similarity for selected genes (no eps needed since norms > 0)
    cos_sims = []
    for i in selected_idxs:
        v1 = X_pred_full[:, i]
        v2 = X_true_full[:, i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / (n1 * n2))

    # Create gene-score dataframe for selected genes (gene names as index)
    df_g = pd.DataFrame(cos_sims, index=selected_genes, columns=["score"])
    df_g.index.name = "gene"

    # Compute additional per-gene benchmarking metrics using predicted vs true columns
    # Extract selected columns for the metrics functions (shape: n_spots x n_selected_genes)
    X_pred_sel = X_pred_full[:, selected_idxs]
    X_true_sel = X_true_full[:, selected_idxs]

    # Use validation metrics implementations which operate column-wise
    try:
        pcc_vals = vm.pearsonr(X_true_sel, X_pred_sel)
        rmse_vals = vm.RMSE(X_true_sel, X_pred_sel)
        js_vals = vm.JS(X_true_sel, X_pred_sel)
        ssim_vals = vm.ssim(X_true_sel, X_pred_sel)
    except Exception:
        # If any metric computation fails, fill with NaNs to avoid breaking the flow
        n_sel = X_pred_sel.shape[1]
        pcc_vals = np.full(n_sel, np.nan)
        rmse_vals = np.full(n_sel, np.nan)
        js_vals = np.full(n_sel, np.nan)
        ssim_vals = np.full(n_sel, np.nan)

    # Add metric columns to dataframe (align by selected_genes order)
    df_g["PCC"] = pcc_vals
    df_g["RMSE"] = rmse_vals
    df_g["JS"] = js_vals
    df_g["SSIM"] = ssim_vals
    # Annotate training/validation flags from spatial var (preferred)
    st_var = datamodule.adata_st[:, selected_genes].var
    df_g["is_training"] = st_var.get("is_training", pd.Series(False, index=st_var.index)).astype(bool)
    df_g["is_validation"] = st_var.get("is_validation", pd.Series(False, index=st_var.index)).astype(bool)


    # Add spatial sparsity - indexes are already aligned
    df_g["sparsity_st"] = datamodule.adata_st[:, selected_genes].var.sparsity
    # Add sc sparsity - inner join indexes
    df_g = df_g.merge(
        pd.DataFrame(datamodule.adata_sc[:, selected_genes].var["sparsity"]),
        left_index=True,
        right_index=True,
    )
    df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
    # Add sparsity difference
    df_g["sparsity_diff"] = df_g["sparsity_st"] - df_g["sparsity_sc"]

    # Add SA statistics
    uns = datamodule.adata_st.uns
    if 'moranI' in uns and 'gearyC' in uns:
        moran_df = uns['moranI'].copy()
        geary_df = uns['gearyC'].copy()

        # Normalize indexes to lowercase
        moran_df.index = moran_df.index.str.lower()
        geary_df.index = geary_df.index.str.lower()

        # Normalize selected genes to lowercase for lookup
        selected_genes_l = [g.lower() for g in selected_genes]

        # Check expected SA columns
        if 'I' not in moran_df.columns or 'C' not in geary_df.columns:
            warnings.warn('SA statistics found but malformed; skipping.')
        else:
            # Align on lowercase gene names
            moran_aligned = moran_df.loc[selected_genes_l, 'I']
            geary_aligned = geary_df.loc[selected_genes_l, 'C']

            df_g['moranI'] = moran_aligned.values
            df_g['gearyC'] = geary_aligned.values

    else:
        warnings.warn('SA statistics not found in .uns; skipping.')


    # Sort scores
    df_g = df_g.sort_values(by="score", ascending=False)

    return df_g