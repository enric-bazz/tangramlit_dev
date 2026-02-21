import numpy as np
import scanpy as sc
import pandas as pd
import warnings
import lightning.pytorch as lp
from lightning.pytorch.loggers import TensorBoardLogger



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
    validation_dict = trainer.validate(model=model, datamodule=datamodule)

    return validation_dict



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


def compare_spatial_geneexp(adata_ge, datamodule):
    """Compare predicted spatial expression against true spatial expression.

    Compares generated spatial gene expression (`adata_ge`) with the true spatial
    expression contained in `datamodule.adata_st`. For each gene present in both
    datasets (and non-zero in both), the function computes:
      - cosine similarity (returned in column ``score``)

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
        ``score``, ``is_training``,
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