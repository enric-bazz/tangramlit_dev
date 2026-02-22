"""
Benchmarking metrics for Tangram. The metrics and scaling procedures are taken from the following paper:
Li, B., Zhang, W., Guo, C. et al. "Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution"
Nat Methods (2022). https://doi.org/10.1038/s41592-022-01480-9.
"""

import numpy as np
import scipy.stats as st

def SSIM(raw, impute, scale='scale_max'):
    """
    Calculate SSIM values between columns of two 2D arrays.
    raw, impute: numpy arrays of shape (n_spots, n_genes)
    Returns: numpy array of shape (n_genes,)
    """
    # optional scaling
    if scale == 'scale_max':
        raw = _scale_max(raw)
        impute = _scale_max(impute)

    if raw.shape[1] != impute.shape[1]:
        raise ValueError("Validation metrics error: raw and impute must have the same number of columns")

    n_genes = raw.shape[1]
    ssim_values = np.zeros(n_genes)

    for j in range(n_genes):
        raw_col = raw[:, j]
        impute_col = impute[:, j]

        M = raw_col.max() if raw_col.max() > impute_col.max() else impute_col.max()

        raw_col_2 = raw_col.reshape(-1, 1)
        impute_col_2 = impute_col.reshape(-1, 1)

        ssim_values[j] = _cal_ssim(raw_col_2, impute_col_2, M)

    return ssim_values

def PCC(raw, impute):
    """
    Calculate Pearson correlation coefficient between corresponding columns
    of two 2D arrays.

    raw, impute: numpy arrays of shape (n_samples, n_genes)
    Returns: numpy array of shape (n_genes,)
    """
    if raw.shape[1] != impute.shape[1]:
        raise ValueError("Validation metrics error: raw and impute must have the same number of columns")

    n_genes = raw.shape[1]
    pearson_values = np.zeros(n_genes)

    for j in range(n_genes):
        raw_col = raw[:, j]
        impute_col = impute[:, j]
        pearson_values[j], _ = st.pearsonr(raw_col, impute_col)

    return pearson_values


def JS(raw, impute, scale='scale_plus'):
    """
    Calculate the Jensen-Shannon divergence between corresponding columns
    of two 2D arrays.

    raw, impute: numpy arrays of shape (n_samples, n_genes)
    Returns: numpy array of shape (n_genes,)
    """
    if scale == 'scale_plus':
        raw = _scale_plus(raw)
        impute = _scale_plus(impute)

    if raw.shape[1] != impute.shape[1]:
        raise ValueError("Validation metrics error: raw and impute must have the same number of columns")

    n_genes = raw.shape[1]
    js_values = np.zeros(n_genes)

    for j in range(n_genes):
        raw_col = raw[:, j]
        impute_col = impute[:, j]

        # Ensure they are proper probability distributions (avoid log(0))
        raw_col = np.clip(raw_col, 1e-12, 1.0)
        impute_col = np.clip(impute_col, 1e-12, 1.0)

        M = 0.5 * (raw_col + impute_col)
        js_values[j] = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)

    return js_values


def RMSE(raw, impute, scale='zscore'):
    """
    Calculate the root mean squared error between corresponding columns
    of two 2D arrays.

    raw, impute: numpy arrays of shape (n_samples, n_genes)
    Returns: numpy array of shape (n_genes,)
    """
    if scale == 'zscore':
        raw = _scale_z_score(raw)
        impute = _scale_z_score(impute)

    if raw.shape[1] != impute.shape[1]:
        raise ValueError("Validation metrics error: raw and impute must have the same number of columns")

    n_genes = raw.shape[1]
    rmse_values = np.zeros(n_genes)

    for j in range(n_genes):
        raw_col = raw[:, j]
        impute_col = impute[:, j]

        diff = raw_col - impute_col
        rmse_values[j] = np.sqrt(np.mean(diff ** 2))

    return rmse_values


def _cal_ssim(im1, im2, M):
    """
    Calculate the SSIM value between two arrays.
    Parameters
    -------
    im1 : array-like, shape (n_samples, 1) or (n_samples, n_genes)
    im2 : array-like, same shape as im1
    M   : float, max value among im1 and im2
    """
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    assert im1.shape == im2.shape
    assert im1.ndim == 2

    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()

    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2

    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    return l12 * c12 * s12


def _scale_max(arr):
    """
    Scale columns by dividing each by its maximum value.
    Input: arr (n_samples, n_genes)
    Output: scaled array (same shape)
    """
    arr = np.asarray(arr, dtype=float)
    max_vals = np.max(arr, axis=0)
    # Avoid division by zero
    max_vals[max_vals == 0] = 1.0
    return arr / max_vals


def _scale_z_score(arr):
    """
    Scale columns by z-score: mean=0, std=1
    Input: arr (n_samples, n_genes)
    Output: scaled array (same shape)
    """
    arr = np.asarray(arr, dtype=float)
    return st.zscore(arr, axis=0, ddof=0)


def _scale_plus(arr):
    """
    Scale columns so they sum to 1 (softmax-like normalization).
    Input: arr (n_samples, n_genes)
    Output: scaled array (same shape)
    """
    arr = np.asarray(arr, dtype=float)
    sums = np.sum(arr, axis=0)
    # Avoid division by zero
    sums[sums == 0] = 1.0
    return arr / sums