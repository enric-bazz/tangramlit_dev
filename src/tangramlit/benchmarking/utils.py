import pandas as pd
import numpy as np

from .metrics import PCC, RMSE, JS, SSIM

def benchmark_mapping(adata_ge, datamodule, df_g):
    """Benchmark predicted spatial expression against true spatial expression.

    Works on top of the output of tangramlit.mapping.utils.compare_spatial_geneexpr() by adding benchmarking metrics.
    For each gene present in both datasets (and non-zero in both), the function computes:
      - Pearson correlation coefficient (``PCC``)
      - Root mean squared error (``RMSE``)
      - Jensen-Shannon divergence (``JS``)
      - Structural Similarity Index (``SSIM``)

    Args:
        adata_ge (AnnData): generated spatial data returned by
            :func:`project_sc_genes_onto_space()` (spots x genes).
        datamodule (LightningDataModule): LightningDataModule containing the
            preprocessed single-cell and spatial data (used for ground-truth and
            sparsity annotations).
        df_g (pandas.DataFrame): output dataframe of 
            tangramlit.mapping.utils.compare_spatial_geneexpr()

    Returns:
        pd.DataFrame: Per-gene metrics indexed by gene name. Added columns include
        ``PCC``, ``RMSE``, ``JS``, ``SSIM``.
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


    # Sort scores
    df_g = df_g.sort_values(by="score", ascending=False)

    return df_g

def aggregate_benchmarking_metrics(df_g: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-gene benchmarking metric values.

    Returns a dataframe with mean, Q1, median, Q3
    for each metric (PCC, RMSE, JS, SSIM).
    """

    cols = ["PCC", "RMSE", "JS", "SSIM"]

    missing = [c for c in cols if c not in df_g.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary = pd.DataFrame({
        "mean": df_g[cols].mean(),
        "q1": df_g[cols].quantile(0.25),
        "median": df_g[cols].quantile(0.5),
        "q3": df_g[cols].quantile(0.75),
    })

    return summary