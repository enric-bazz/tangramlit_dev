import warnings
import pandas as pd
import numpy as np

from .metrics import PCC, RMSE, JS, SSIM

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
        pcc_vals = PCC(X_true_sel, X_pred_sel)
        rmse_vals = RMSE(X_true_sel, X_pred_sel)
        js_vals = JS(X_true_sel, X_pred_sel)
        ssim_vals = SSIM(X_true_sel, X_pred_sel)
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