"""
Mapping functions for Tangram using the Lightning framework.
"""
import pandas as pd
import sklearn
from anndata import AnnData
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold

from . import utils as ut
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
    if lambda_g1 > 0:
        raise ValueError("lambda_g1 cannot be 0.")

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

    # Filter inputs
    if filter:
        if not all([lambda_f_reg, lambda_count]):
            raise ValueError(
                "lambda_f_reg and lambda_count must be specified if cell filter is active."
            )
        if not target_count:
            target_count = adata_st.shape[0]
            logging.info(f"target_count missing from input is set to adata_st.shape[0] = {target_count}")

    # CT islands enforcement
    if lambda_ct_islands > 0:
        if cluster_label is None or cluster_label not in adata_sc.obs.keys():
            raise ValueError(
                "cluster_label must be specified for the cell type island loss term."
            )

    # Check spatial coordinates for refined mode
    if 'spatial' not in adata_st.obsm.keys():
        raise ValueError("'spatial' key is missing in adata_st.obsm.")
        
    # Check for NaN values in spatial coordinates (squidpy graph does not work otherwise)
    if np.any(np.isnan(adata_st.obsm['spatial'])):
        # Get indices of rows without NaN values
        valid_spots = ~np.any(np.isnan(adata_st.obsm['spatial']), axis=1)
        n_invalid = np.sum(~valid_spots)
        
        logging.warning(f"Found {n_invalid} spots in adata_st with NaN coordinates. These spots will be removed.")
        
        # Remove spots with NaN coordinates
        adata_st._inplace_subset_obs(valid_spots)

    # Create hyperparameters dictionary for all next calls
    hyperparameters = {}
    # Add filter
    hyperparameters['filter'] = filter

    # Add learning rate and n_epochs
    hyperparameters['learning_rate'] = learning_rate
    hyperparameters['num_epochs'] = num_epochs
    # Add RNG seed
    hyperparameters['random_state'] = random_state

    # Loss
    loss_terms = {
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
        "cluster_label": cluster_label,  # labels column
    }
    hyperparameters.update(loss_terms)
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
):
    """
    Map single cell data (`adata_sc`) on spatial data (`adata_sp`).

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

    # Call the data module to retrieve batch size
    data = MyDataModule(adata_sc=adata_sc,
                        adata_st=adata_st,
                        input_genes=input_genes,
                        train_genes_names=train_genes_names,
                        val_genes_names=val_genes_names,
                        )
    # NOTE: Need to create the datamodule instance before computing spatial weights, for which the spatial neighborhood graph is required

    # Compute spatial weights for refined mode
    voxel_weights, neighborhood_filter, ct_encode, spatial_weights = None, None, None, None
    if lambda_neighborhood_g1 > 0:
        voxel_weights = compute_spatial_weights(adata_st, standardized=True, self_inclusion=True)
    if lambda_ct_islands > 0:
        neighborhood_filter = compute_spatial_weights(adata_st, standardized=False, self_inclusion=False)
        ct_encode = ut.one_hot_encoding(adata_sc.obs[cluster_label]).values
    if lambda_moran > 0 or lambda_geary > 0:
        spatial_weights = compute_spatial_weights(adata_st, standardized=True, self_inclusion=False)
    if lambda_getis_ord > 0:
        spatial_weights = compute_spatial_weights(adata_st, standardized=False, self_inclusion=True)

    # Add to hyperparameters dictionary
    hyperparameters['spatial_weights'] = spatial_weights
    hyperparameters['neighborhood_filter'] = neighborhood_filter
    hyperparameters['ct_encode'] = ct_encode
    hyperparameters['voxel_weights'] = voxel_weights

    # NOTE: This step is implemented here as it requires both the lambda coefficientss and the anndata objects.
    # It is possible to implement it in the LightningDataModule, but it would require passing the lambda values
    # to the LightningDataModule and then to the LightningModule. This would require modifying the LightningDataModule
    # to accept lambda values as input.
    # The same goes for the PytorchLightningModule that would, instead, require the anndata objects as inputs.

    # Initialize the model
    model = MapperLightning(**hyperparameters)

    # Customize ModelCheckpoint callback to avoid memory blow-up
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # dirpath="checkpoints/",
        # filename="best-checkpoint",
        save_top_k=0,
        # monitor="val_loss",
        # mode="min",
        save_weights_only=True
    )

    # Set early stopper
    early_stop = EarlyStopping(
        monitor="main_loss",  # monitor training score
        min_delta=0.001,  # score minimum improvement
        patience=10,
        verbose=False,
        mode="max",
        check_on_train_epoch_end=True,
    )

    # Create TB logger for training
    train_logger = TensorBoardLogger("tb_logs", name="train")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # log_every_n_steps=print_each,
        logger=train_logger,
        callbacks=[EpochProgressBar(), early_stop],
        # callbacks=[checkpoint_callback],
        enable_checkpointing=False,  # disable lightning_checkpoints
        enable_progress_bar=True,
        check_val_every_n_epoch=200,
        #num_sanity_val_steps=0,  # skip sanity check
        #limit_val_batches=0,  # disables validation during fit
    )

    # Train the model
    trainer.fit(model, datamodule=data)

    # Get the final mapping matrix
    with torch.no_grad():
        if model.hparams.filter:
            mapping, _, filter_probs = model()  # Unpack values (skip filtered M matrix)
            final_mapping = mapping.cpu().numpy()  # Turn into numpy array
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
        'ct_island_term': model.loss_history['ct_island_term'],
        'getis_ord_term': model.loss_history['getis_ord_term'],
        'moran_term': model.loss_history['moran_term'],
        'geary_term': model.loss_history['geary_term'],
    }
    # Add filter terms only if filter was used
    if model.hparams.filter:
        filter_terms = {
            'count_reg': model.loss_history['count_reg'],
            'lambda_f_reg': model.loss_history['lambda_f_reg']
        }
        training_history.update(filter_terms)
        # Include filter training history logs
        adata_map.uns['filter_history'] = {
            'filter_values': model.filter_history['filter_values'],
            'n_cells': model.filter_history['n_cells']
        }
    # Store training history in adata_map.uns['training_history']
    adata_map.uns['training_history'] = training_history

    return adata_map, model, data


def compute_spatial_weights(adata_st, standardized, self_inclusion):
    """
        Compute spatial weights matrix.
        Contains either row-standardized distances or binary adjacencies. Can include self or not (1 or 0 diagonal).
    """
    if standardized:
        connectivities = adata_st.obsp['spatial_connectivities'].copy()
        distances = adata_st.obsp['spatial_distances'].copy()

        # Row-normalize distances
        distances_norm = sklearn.preprocessing.normalize(distances, norm="l1", axis=1, copy=False)

        # Mask normalized distances with connectivity (keep only neighbor links)
        weighted_adj = connectivities.multiply(distances_norm)

        # Make dense
        spatial_weights = weighted_adj.todense()
    else:
        spatial_weights = adata_st.obsp['spatial_connectivities'].todense()
    if self_inclusion:
        spatial_weights += np.eye(spatial_weights.shape[0])

    return spatial_weights

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
    # Make sc expression matrix dense
    if hasattr(datamodule.adata_sc.X, "toarray"):
        datamodule.adata_sc.X = datamodule.adata_sc.X.toarray()
    # Project sc expression matrix onto space
    X_space = adata_map.X.T @ datamodule.adata_sc.X
    # Create AnnData object with the projected spatial gene expression
    adata_ge = sc.AnnData(
        X=X_space,  # projected sc expression profiles
        obs=adata_map.var,  # spatial data spot IDs
        var=datamodule.adata_sc.var,  # all sc gene names
        uns=datamodule.adata_sc.uns,  # unstructureed fields of sc data 
    )
    # Annotate training genes in adata_ge based on spatial data annotation
    adata_ge.var["is_training"] = adata_ge.var.index.isin(datamodule.adata_st.var.index[datamodule.adata_st.var["is_training"]])
    
    return adata_ge

def compare_spatial_gene_exp(adata_ge, datamodule, input_genes=None):
    """ 
        Compares generated spatial data with the true spatial data.
        The main issue that arises when comparing the expression profiles of predicted and true spatial data is that non-expressed
        genes (i.e. all-zeros), that are excluded from the training genes, cannot be used with the cossim metric directly.
        To overcome this, non-expressed genes counts are set to an arbitrarily small value 'eps' to avoid zero division errors.
        The resulting similarity is zero.
    
        Args:
            adata_ge (AnnData): generated spatial data returned by `project_sc_genes_onto_space()`.
            datamodule (LightningDataModule): LightningDataModule object containing the preprocessed single cell and spatial data (sparsity annotation).
            genes (list): Optional. When passed, returned output will be subset on the list of genes. Default is None.
    
        Returns:
            Pandas Dataframe: a dataframe with columns: 'score', 'is_training', 'sparsity_st'(spatial data sparsity),
                'sparsity_sc'(single cell data sparsity), 'sparsity_diff'(spatial sparsity - single cell sparsity).
    """
    # Get all overlapping genes (training and not)
    if input_genes is None:
        overlap_genes = datamodule.adata_st.uns["overlap_genes"]
    else:
        overlap_genes = input_genes

    # Annotate cosine similarity of each overlapping gene
    cos_sims = []
    # Predicted spatial expression matrix
    if hasattr(adata_ge.X, "toarray"):
        X_pred = adata_ge[:, overlap_genes].X.toarray()
    else:
        X_pred = adata_ge[:, overlap_genes].X
    # True spatial expression matrix
    if hasattr(datamodule.adata_st.X, "toarray"):
        X_true = datamodule.adata_st[:, overlap_genes].X.toarray()
    else:
        X_true = datamodule.adata_st[:, overlap_genes].X
    # Compute row-wise cossim
    eps = 1e-12
    for v1, v2 in zip(X_pred.T, X_true.T):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / max(n1 * n2, eps))
    # Create gene-score dataframe
    df_g = pd.DataFrame(cos_sims, overlap_genes, columns=["score"])
    for adata in [adata_ge, datamodule.adata_st]:
        if "is_training" in adata.var.keys():
            df_g["is_training"] = adata.var.is_training

    # Add spatial sparsity - indexes are already aligned
    df_g["sparsity_st"] = datamodule.adata_st[:, overlap_genes].var.sparsity
    # Add sc sparsity - inner join indexes
    df_g = df_g.merge(
        pd.DataFrame(datamodule.adata_sc[:, overlap_genes].var["sparsity"]),
        left_index=True,
        right_index=True,
    )
    df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
    # Add sparsity difference
    df_g["sparsity_diff"] = df_g["sparsity_st"] - df_g["sparsity_sc"]

    # Sort scores
    df_g = df_g.sort_values(by="score", ascending=False)

    return df_g


def validate_mapping_experiment(model, datamodule):
    """
    Validate mapping experiment. The model must be already trained --> Run only after map_cells_to_space().

    Returns:
        validation_results (dict): Dictionary containing validation results with
        dict_keys(['val_AUC', 'val_SSIM', 'val_PCC', 'val_RMSE', 'val_JS']).
        Note that when called as 'results = validate_mapping_experiment(model, datamodule)', results is a list with
        element results[0] containing the dictionary.
    """
    val_logger = TensorBoardLogger("tb_logs", name="val")
    trainer = pl.Trainer(
        logger=val_logger,
        enable_progress_bar=False,  # disable progress bar
        num_sanity_val_steps=0,  # skip sanity check
    )
    validation_results = trainer.validate(model=model, datamodule=datamodule)

    return validation_results

def get_cv_data(genes_list, k=10):
    """
        Generates a pair of training/test gene indexes for cross-validation.

        Args:
            genes_list (list): List of genes to be used for cross-validation.
            k (int): Number of folds for k-folds cross-validation. Default is 10.

        Yields:
            tuple: list of train_genes, list of test_genes
    """
    genes_array = np.array(genes_list)

    cv = KFold(n_splits = k)

    for train_idx, test_idx in cv.split(genes_array):
        train_genes = list(genes_array[train_idx])
        test_genes = list(genes_array[test_idx])
        yield train_genes, test_genes



def cross_validate_mapping(
        adata_sc,
        adata_st,
        filter=False,
        learning_rate=0.1,
        num_epochs=1000,
        k=10,
        input_genes=None,
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
        random_state=None,
):
    """
    Executes genes set cross-validation using Lightning mapper.
    The genes set must be a subset of the shared genes set: verified with the validate_mapping_inputs() call inside map_cells_to_space().

    Args:
        adata_sc (AnnData): single cell data
        adata_st(AnnData): gene spatial data
        filter (bool): Whether cell filtering is active or not.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        k (int): Number of folds for k-folds cross-validation. Default is 10.
        input_genes (list): Optional. Set of genes to be used for cross-validation.
        lambda_d (float): Optional. Strength of density regularizer. Default is 0.
        lambda_g1 (float): Optional. Strength of Tangram loss function. Default is 1.
        lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when filter is True.
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when filter is True. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        lambda_sparsity_g1 (float): Optional. Strength of sparsity regularizer. Default is 0.
        lambda_neighborhood_g1 (float): Optional. Strength of neighborhood regularizer. Default is 0.
        lambda_getis_ord (float): Optional. Strength of Getis-Ord regularizer. Default is 0.
        lambda_moran (float): Optional. Strength of Moran regularizer. Default is 0.
        lambda_geary (float): Optional. Strength of Geary regularizer. Default is 0.
        lambda_ct_islands (float): Optional. Strength of ct islands enforcement. Default is 0.
        cluster_label (str): Optional. Field in `adata_sc.obs` used for aggregating single cell data. Only valid for lambda_ct-islands > 0. Default is None.
        random_state (int): Optional. pass an int to reproduce training. Default is None.

    Returns:
        cv_metrics (dict): Dictionary containing average metric scores across all folds
    """


    curr_cv_set = 1

    # Init fold metrics dictionary
    cv_metrics = {}

    # Check n folds
    if k > 0:
        length = k
    else:
        raise ValueError("Invalid number of folds. Please enter a positive integer greater than 0.")

    for train_genes, test_genes in get_cv_data(genes_list=input_genes, k=k):

        print(f"Enter loop with test_genes:{test_genes}")
        # Train mapper
        adata_map, mapper, datamodule = map_cells_to_space(
            adata_sc=adata_sc,
            adata_st=adata_st,
            input_genes=input_genes,  # input genes list
            train_genes_names=train_genes,  # training genes of current fold
            val_genes_names=test_genes,  # validation genes of current fold
            mode=mode,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            lambda_d=lambda_d,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_r=lambda_r,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_count=lambda_count,
            lambda_f_reg=lambda_f_reg,
            target_count=target_count,
            lambda_sparsity_g1=lambda_sparsity_g1,
            lambda_neighborhood_g1=lambda_neighborhood_g1,
            lambda_getis_ord=lambda_getis_ord,
            lambda_moran=lambda_moran,
            lambda_geary=lambda_geary,
            lambda_ct_islands=lambda_ct_islands,
            cluster_label=cluster_label,
            random_state=random_state,
        )

        # Compute current fold metrics
        fold_metrics = validate_mapping_experiment(model=mapper, datamodule=datamodule)[0]
        # Update cv dictionary
        for key, value in fold_metrics.items():
            if key not in cv_metrics:
                cv_metrics[key] = []  # first time, make a list
            cv_metrics[key].append(value)  # append new fold value


    # Calculate average metrics across folds
    """cv_metrics = {}
    for metric in metrics:
        temp_arr = np.zeros(len(fold_metrics[metric]))  # shape = (k,)
        for fold in range(len(fold_metrics[metric])):
            temp_arr[fold] = np.mean(fold_metrics[metric][fold])  # scalar
        cv_metrics[metric] = np.array(temp_arr, dtype='float32').mean().item()  # assing metric mean over folds"""

    return cv_metrics

# TODO: refactor so that validation genes can be absent from the spatial data (not a subset of shared genes but only of adata_sc genes)

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
    If filter is active the final filters matrix is also returned.
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
    datamodule = MyDataModule(adata_sc=adata_sc,
                        adata_st=adata_st,
                        #input_genes=input_genes,
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
        trainer = pl.Trainer(
            max_epochs=config['num_epochs'],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_val_batches=0  # disables validation during fit
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
