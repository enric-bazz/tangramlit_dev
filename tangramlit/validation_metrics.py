"""
Validation metrics for Tangram.
The main validation metric presented in the original Tangram paper is the AUC in the (sparsity, score) plane.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import sklearn
import torch
from sklearn.metrics import auc



def poly2_auc(gv_scores, gene_sparsity, pol_deg=2, plot_auc=False):
    """
    Compute Tangram most important evaluation metric.
    Fit a 2nd-degree polynomial between gv_scores and gene_sparsity,
    clip to [0,1], and return the area under the curve (AUC).

    Args:
        gv_scores : array-like or torch.Tensor
            Cosine similarity scores per gene.
        gene_sparsity : array-like or torch.Tensor
            Gene sparsity values per gene (0-1).
        pol_deg : int
            Degree of polynomial (default 2).
        plot_auc (bool): Wether to plot the polynomial fit in the (sparsity, score) plane or not. Default is False.

    Returns:
        auc_score (float): Area under the polynomial curve over x in [0,1].
        auc_coordinates (tuple): AUC fitted coordinates and raw coordinates (test_score vs. sparsity_st coordinates)
        plot: polyfit curve and genes (sparsity, score) scatter plot
    """

    # Convert to numpy arrays
    xs = np.array(gv_scores).flatten()
    ys = np.array(gene_sparsity).flatten()

    # Fit polynomial
    pol_cs = np.polyfit(xs, ys, pol_deg)
    pol = np.poly1d(pol_cs)

    # Sample polynomial on [0,1]
    pol_xs = np.linspace(0, 1, 50)
    pol_ys = pol(pol_xs)

    # Clip values to [0,1]
    pol_ys = np.clip(pol_ys, 0, 1)

    # Include real roots where y=0 inside [0,1]
    roots = pol.r
    for r in roots:
        if np.isreal(r) and 0 <= r.real <= 1:
            pol_xs = np.append(pol_xs, r.real)
            pol_ys = np.append(pol_ys, 0)

    # Sort x values for proper integration
    sort_idx = np.argsort(pol_xs)
    pol_xs = pol_xs[sort_idx]
    pol_ys = pol_ys[sort_idx]

    # Compute AUC
    auc_score = auc(pol_xs, pol_ys)

    # Coordinates
    auc_coordinates = ((pol_xs, pol_ys), (xs, ys))

    if plot_auc:
        fig = plt.figure()
        plt.figure(figsize=(6, 5))

        plt.plot(pol_xs, pol_ys, c='r')
        sns.scatterplot(x=xs, y=ys, alpha=0.5, edgecolors='face')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.gca().set_aspect(.5)
        plt.xlabel('score')
        plt.ylabel('spatial sparsity')
        plt.tick_params(axis='both', labelsize=8)
        plt.title('Prediction on validation transcriptome')

        textstr = 'auc_score={}'.format(np.round(auc_score, 3))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        # place a text box in upper left in axes coords
        plt.text(0.03, 0.1, textstr, fontsize=11, verticalalignment='top', bbox=props)
        plt.show()

    return float(auc_score), auc_coordinates



"""
Validation metrics for Tangram benchmarking. The metrics and scaling procedures are taken from the following paper:
Li, B., Zhang, W., Guo, C. et al. Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution.
Nat Methods (2022). https://doi.org/10.1038/s41592-022-01480-9.
"""


def ssim(raw, impute, scale='scale_max'):
    """
    Calculate SSIM values between columns of two 2D arrays.
    raw, impute: numpy arrays of shape (n_spots, n_genes)
    Returns: numpy array of shape (n_genes,)
    """
    # optional scaling
    if scale == 'scale_max':
        raw = scale_max(raw)
        impute = scale_max(impute)

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

        ssim_values[j] = cal_ssim(raw_col_2, impute_col_2, M)

    return ssim_values

def pearsonr(raw, impute):
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
        raw = scale_plus(raw)
        impute = scale_plus(impute)

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
        raw = scale_z_score(raw)
        impute = scale_z_score(impute)

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


def cal_ssim(im1, im2, M):
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


def scale_max(arr):
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


def scale_z_score(arr):
    """
    Scale columns by z-score: mean=0, std=1
    Input: arr (n_samples, n_genes)
    Output: scaled array (same shape)
    """
    arr = np.asarray(arr, dtype=float)
    return st.zscore(arr, axis=0, ddof=0)


def scale_plus(arr):
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


""" 
Validation metrics for Tangram benchmarking. The metrics are used to validate consistency across different runs and against ground truth mapping.
Definitions are taken from the paper:
Refinement Strategies for Tangram for Reliable Single-Cell to Spatial Mapping. Stahl, et al (2025), https://doi.org/10.1101/2025.01.27.634996.
"""

# METRICS FOR GROUND TRUTH COMPARISONS (might require annotated spatial data, real or synthetic)

def cosine_similarity(true_values, pred_values, axis):
    """
    Compute the cosine similarity between true and predicted values
    Args:
        true_values (Array): Ground truth (k,j)
        pred_values (Array): Predicted values (r,k,j)
        axis (int): Axis where to perform the comparison, can be 1 or 2
    Returns:
        Array: Cosine similarity values (r,k) o r(r,j)
    Example:
        Gene expression prediction correctness for each gene along the spots: n_runs x n_genes x n_spots => n_runs x n_genes
        Gene expression prediction correctness for each spot along the genes: n_runs x n_genes x n_spots => n_runs x n_spots
    """
    return np.array(torch.nn.functional.cosine_similarity(torch.Tensor(pred_values),
                                                          torch.Tensor(true_values),
                                                          dim=axis))


def categorical_cross_entropy(true_probs, pred_probs_cube):
    """
    Compute the categorical cross-entropy between true and predicted probabilities cube along the last axis
    Args:
        true_probs (Array): Ground truth (i,j)
        pred_probs_cube (Array): Predicted values (r,i,j)
    Returns:
        Array: Cross-entropy values (r,i)
    Example:
        Cell mapping correctness for each cell along the spots: n_runs x n_cells x n_spots => n_runs x n_cells
    """
    entropy = []
    for run in range(pred_probs_cube.shape[0]):
        tmp = []
        for i in range(pred_probs_cube.shape[1]):
            tmp.append(sklearn.metrics.log_loss([true_probs[i].argmax()], np.array([pred_probs_cube[run, i]]),
                                                labels=range(true_probs.shape[1]), normalize=False))
        entropy.append(tmp)
    return np.array(entropy) / np.array(entropy).mean(axis=0).max()


def multi_label_categorical_cross_entropy(true_probs, pred_probs_cube):
    """
    Compute the multi-label categorical cross-entropy between true and predicted probabilities cube along the last axis
    Args:
        true_probs (Array): Ground truth (i,j)
        pred_probs_cube (Array): Predicted values (r,i,j)
    Returns:
        Array: Multi-label cross-entropy values (r,i)
    Example:
        Cell type mapping correctness for each cell type along the spots: n_runs x n_celltypes x n_spots => n_runs x n_celltypes
    """
    entropy = []
    for run in range(pred_probs_cube.shape[0]):
        tmp = []
        for i in range(pred_probs_cube.shape[1]):
            tmp.append(sklearn.metrics.log_loss(np.array([true_probs[i], 1 - true_probs[i]]).argmax(axis=0),
                                                np.array([pred_probs_cube[run, i], 1 - pred_probs_cube[run, i]]).T,
                                                labels=[0, 1], normalize=True))
        entropy.append(tmp)
    return np.array(entropy) / np.array(entropy).mean(axis=0).max()


# METRICS FOR TRAINING RUN COMPARISONS

def pearson_corr(cube):
    """
    Compute the pearson correlation for the first axis
    Args:
        cube (Array): Values (r,n,j)
    Returns:
        Array: All pairwise Pearson correlations (r x r)
    Example:
        Cell (type) mapping or gene expression prediction consistency: n_runs x n_genes/cell(type)s x n_spots => n_runpairs
    """
    idx = np.tril_indices(cube.shape[0], -1)
    return np.corrcoef(np.reshape(cube, (cube.shape[0], -1)))[idx]


def pearson_corr_over_axis(cube, axis):
    """
    Compute the pearson correlation for a given axis, averaged across all pairwise correlation from the first axis
    Args:
        cube (Array): Values (r,n,j)
        axis (int): Axis, can be 1 or 2
    Returns:
        Array: Mean Pearson correlations along a specific axis (n) or (j)
    Example:
        Cell (type) mapping or gene expression prediction consistency: n_runs x n_genes/cell(type)s x n_spots => n_genes/cell(type)s or n_spots
    """
    all_pearsons = []
    idx = np.tril_indices(cube.shape[0], -1)
    if axis == 1:
        for i in range(cube.shape[1]):
            all_pearsons.append(np.corrcoef(cube[:, i, :])[idx].mean())
    else:  # axis == 2
        for i in range(cube.shape[2]):
            all_pearsons.append(np.corrcoef(cube[:, :, i])[idx].mean())
    return np.array(all_pearsons)


def vote_entropy(pred_probs_cube):
    """
    Compute the normalized vote entropy across the last axis
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Vote entropy values (r,i)
    Example:
        Cell mapping agreement: n_runs x n_cells x n_spots => n_runs x n_cells
    """
    votes_encoded = np.zeros(pred_probs_cube.shape)
    votes = pred_probs_cube.argmax(axis=2)
    for run in range(pred_probs_cube.shape[0]):
        votes_encoded[run, np.arange(pred_probs_cube.shape[1]), votes[run]] = 1
    return st.entropy(votes_encoded.mean(axis=0), axis=1) / np.log(pred_probs_cube.shape[2])


def multi_label_vote_entropy(pred_probs_cube):
    """
    Compute the normalized multi-label vote entropy
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Multi-label vote entropy values (r,i,j)
    Example:
        Cell type mapping agreement: n_runs x n_celltypes x n_spots => n_runs x n_celltypes x n_spots
    """
    votes_encoded = np.round(pred_probs_cube)
    return st.entropy(np.array([votes_encoded.mean(axis=0), 1 - votes_encoded.mean(axis=0)]), axis=0) / np.log(
        2)


def consensus_entropy(pred_probs_cube):
    """
    Compute the normalized consensus entropy across the last axis
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Consensus entropy values (r,i)
    Example:
        Cell mapping certainty: n_runs x n_cells x n_spots => n_runs x n_cells
    """
    consensus_mapping = pred_probs_cube.mean(axis=0)
    return st.entropy(consensus_mapping, axis=1) / np.log(pred_probs_cube.shape[2])


def multi_label_consensus_entropy(pred_probs_cube):
    """
    Compute the normalized multi-label consensus entropy
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Multi-label consensus entropy values (r,i,j)
    Example:
        Cell type mapping certainty: n_runs x n_celltypes x n_spots => n_runs x n_celltypes x n_spots
    """
    consensus_mapping = pred_probs_cube.mean(axis=0)
    return st.entropy(np.array([consensus_mapping, 1 - consensus_mapping]), axis=0) / np.log(2)