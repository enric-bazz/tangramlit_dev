import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import classification_report, accuracy_score

from . import utils as ut

## HEAD TO HEAD MODEL COMPARISON  ##
"""
Utilities for models comparison. Specifically, bulk metrics over all data.
"""

# Plot losses side by side for comparison
def compare_loss_trajectories(hist1, hist2, key='total_loss'):
    plt.figure(figsize=(12, 6))
    plt.plot(hist1[key], label='Original', alpha=0.7)
    plt.plot(hist2[key], label='Lightning', alpha=0.7)
    plt.title(f'Comparison of {key}')
    plt.xlabel('Epoch')
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.show()


# Compare intermediate states
def analyze_mapping_evolution(matrix1, matrix2):
    # Compute correlation
    correlation = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]

    # Compute cosine similarity
    cos_sim = np.dot(matrix1.flatten(), matrix2.flatten()) / \
              (np.linalg.norm(matrix1.flatten()) * np.linalg.norm(matrix2.flatten()))

    # Find largest differences
    diff = np.abs(matrix1 - matrix2)
    max_diff_pos = np.unravel_index(np.argmax(diff), diff.shape)

    return {
        'correlation': correlation,
        'cosine_similarity': cos_sim,
        'max_diff': np.max(diff),
        'max_diff_position': max_diff_pos,
        'max_diff_values': (matrix1[max_diff_pos], matrix2[max_diff_pos])
    }


# Analyze distribution of values
def plot_mapping_distributions(matrix1, matrix2, title="Distribution of Mapping Values"):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(matrix1.flatten(), bins=50, alpha=0.7, label='Original')
    plt.hist(matrix2.flatten(), bins=50, alpha=0.7, label='Lightning')
    plt.title(title)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(matrix1.flatten(), matrix2.flatten(), alpha=0.1)
    plt.plot([0, 1], [0, 1], 'r--')  # diagonal line for reference
    plt.xlabel('Original Values')
    plt.ylabel('Lightning Values')
    plt.title('Value Correlation')

    plt.tight_layout()
    plt.show()


# Analyze sparsity patterns
def compare_sparsity(matrix1, matrix2, threshold=1e-5):
    sparse1 = (matrix1 > threshold).astype(float)
    sparse2 = (matrix2 > threshold).astype(float)

    agreement = np.sum(sparse1 == sparse2) / sparse1.size
    diff_positions = np.where(sparse1 != sparse2)

    return {
        'sparsity_agreement': agreement,
        'different_positions': list(zip(diff_positions[0], diff_positions[1]))[:5]  # first 5 differences
    }


def compare_cell_choices(ad_map, ad_map_lt):
    """
    Compare how similarly the two models choose cells by analyzing F_out values.

    Parameters:
    -----------
    ad_map : AnnData
        Result from the original mapping
    ad_map_lt : AnnData
        Result from the lightning mapping

    Returns:
    --------
    dict
        Dictionary containing comparison metrics
    """
    # Get F_out from both models
    f_probs_original = ad_map.obs['F_out'].to_numpy()
    f_probs_lightning = ad_map_lt.obs['F_out'].to_numpy()

    if f_probs_original is None or f_probs_lightning is None:
        return {"error": "F_out not found in one or both models"}

    # Calculate various similarity metrics
    correlation = np.corrcoef(f_probs_original, f_probs_lightning)[0, 1]

    # Calculate cosine similarity between filters
    cos_sim = np.dot(f_probs_original, f_probs_lightning) / \
              (np.linalg.norm(f_probs_original) * np.linalg.norm(f_probs_lightning))

    # Calculate absolute differences
    diff = np.abs(f_probs_original - f_probs_lightning)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)


    results = {
        'correlation': correlation,
        'cosine_similarity': cos_sim,
        'max_difference': max_diff,
        'mean_difference': mean_diff,
    }

    # Visualize the comparison
    plt.figure(figsize=(15, 5))

    # Plot 1: Scatter plot of F_out values
    plt.subplot(131)
    plt.scatter(f_probs_original, f_probs_lightning, alpha=0.1)
    plt.plot([0, 1], [0, 1], 'r--')  # diagonal line for reference
    plt.xlabel('Original F_out')
    plt.ylabel('Lightning F_out')
    plt.title('F_out Correlation')

    # Plot 2: Distribution comparison
    plt.subplot(132)
    plt.hist(f_probs_original, bins=50, alpha=0.5, label='Original')
    plt.hist(f_probs_lightning, bins=50, alpha=0.5, label='Lightning')
    plt.xlabel('F_out values')
    plt.ylabel('Frequency')
    plt.title('Distribution of F_out')
    plt.legend()

    # Plot 3: Difference plot
    plt.subplot(133)
    plt.plot(diff, alpha=0.7)
    plt.xlabel('Cell index')
    plt.ylabel('Absolute difference')
    plt.title('F_out Differences')

    plt.tight_layout()
    plt.show()

    return results

def compute_filter_corr(adata_map, plot=True, plot_regression=False):
    """
    Compute correlation of initial and final filter values
    and optionally plot scatter with regression line.

    Args:
        adata_map: adata object output of map_cells_to_space(). Must contain adata_map.uns['filter_history']
        plot (bool): If True, show scatterplot with regression line.
    """

    if "filter_history" not in adata_map.uns.keys():
        raise ValueError("Missing filter history in mapped input data.")
    # Retrieve filter history
    filter_history = adata_map.uns['filter_history']['filter_values']  # shape = (n_epochs, n_cells)

    # Retrieve initial and final filter values
    filter_init = filter_history[0]
    filter_final = filter_history[-1]

    # Compute correlation
    filter_corr = np.corrcoef(filter_init, filter_final)[0, 1]
    print(f"Pearson correlation coefficient of filter values: {filter_corr}")

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(filter_init, filter_final, alpha=0.5, color="blue", label="Cells")

        if plot_regression:
            # Fit regression line
            slope, intercept, _, _, _ = linregress(filter_init, filter_final)
            x_vals = np.array([filter_init.min(), filter_init.max()])
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, color="red", lw=2, label="Fit")

        plt.xlabel("Initial filter values")
        plt.ylabel("Final filter values")
        plt.title(f"Correlation = {filter_corr:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return filter_corr

### DETERMINISTIC MAPPING EVALUATION ###

# NOTE: Tangram performs an asymmetric cell to spot mapping in the sense that it estimates the probability distribution of
# each cell over spots, but conversely computes just a soft assignment/cell composition of each spot.
# This means that one hot encoding with max() over the rows corresponds to a deterministic mapping of each cell onto a single
# spot (based on its probability density function), while the same over the columns translates into a hard assignment
# of a cell for each spot that is not based on a truly probability distribution, but rather a composition vector.

def get_cell_spot_pair(adata_map, filter=False, threshold=0.5):
    """
    Perform deterministic mapping of cells to spots from the cells persepctive, i.e. each cell distribution is collapsed
    into the max value to deterministically map it into a spot.

    Args:
        adata_map: Mapping object output of map_cells_to_space
        filter (bool): Whether cell filtering is performed. Default: False.
        threshold (float): Threshold value for cell filtering.
        NOTE: Does not guarantee that all spots are assigned to at leas one cell (not feasible for spatial reconstruction).

    Returns:
        Dataframe with shape (n_cells/n_filtered_cells, 2): each row = (spot index, cell index).
    """
    # Input check
    if filter and "F_out" not in adata_map.obs.keys():
        raise ValueError("Missing final filter in mapped input data.")

    ### Deterministic mapping
    # Store mapping matrix
    mapping_matrix = adata_map.X  # shape (n_cells, n_spots)
    if not isinstance(mapping_matrix, np.ndarray):
        mapping_matrix = mapping_matrix.toarray()  # in case it's sparse
    if filter:
        # Filter out non-selected cells
        deterministic_filter = adata_map.obs['F_out'] > threshold  # shape (n_cells,)
        mapping_matrix = mapping_matrix[deterministic_filter, :]  # shape (n_filtered_cells, n_spots)
        cell_index = adata_map.obs_names[deterministic_filter]  # keep only filtered cell names
    else:
        cell_index = adata_map.obs_names
    # Collapse cell mappings to spot corresponding to max probability
    spot_number = np.argmax(mapping_matrix, axis=1)  # shape (n_cells,)
    spot_index = adata_map.var_names[spot_number]  # shape (n_cells,)

    # cell-spot pairs dataframe
    pairs_df = pd.DataFrame({
        "spot index": spot_index,
        "cell index": cell_index,
    }).reset_index(drop=True)

    return pairs_df

def get_spot_cell_pair(adata_map, filter=False, threshold=0.5):
    """
    Perform deterministic mapping of cells to spots from the spots perspective, i.e. each spot get "deterministically"
    assigned to the cell with the largest probability over it, aka its main component.

    Args:
        adata_map: Mapping object output of map_cells_to_space
        filter (bool): Whether cell filtering is performed. Default: False.
        threshold (float): Threshold value for cell filtering.

    Returns:
        DataFrame with shape (n_spots, 2): each row = (spot index, cell index).
    """
    # Input check
    if filter and "F_out" not in adata_map.obs.keys():
        raise ValueError("Missing final filter in mapped input data.")

    # Store mapping matrix
    mapping_matrix = adata_map.X  # shape (n_cells, n_spots)
    if not isinstance(mapping_matrix, np.ndarray):
        mapping_matrix = mapping_matrix.toarray()  # in case it's sparse

    # Handle cell filtering
    if filter:
        deterministic_filter = adata_map.obs['F_out'] > threshold  # shape (n_cells,)
        mapping_matrix = mapping_matrix[deterministic_filter, :]  # shape (n_filtered_cells, n_spots)
        obs_names = adata_map.obs_names[deterministic_filter]  # filtered cell names
    else:
        obs_names = adata_map.obs_names

    # Collapse spot mappings: for each spot (column), find cell with max probability
    cell_number = np.argmax(mapping_matrix, axis=0)  # shape (n_spots,)
    cell_index = obs_names[cell_number]  # map indices to cell names
    spot_index = adata_map.var_names  # all spots (always length n_spots)

    # Spot-cell pairs dataframe
    pairs_df = pd.DataFrame({
        "spot index": spot_index,
        "cell index": cell_index
    }).reset_index(drop=True)

    return pairs_df

def count_deterministic_mapping_matches(cells_to_spots_df, spots_to_cells_df):
    """
    Count the number of matches between a cells to spot deterministic mapping and a spots to cells deterministic mapping.
    This is implemented from the spots perspective. i.e. for each spot:
    - get the assigned cell id from spots_ti_cells_df
    - find the cell in cells_to_spots_df with the same cell id
    - get the assigned spot id from cells_to_spots_df
    - if it matches the spot id, increment the counter

    Args:
        cells_to_spots_df: DataFrame with shape (n_cells, 2) containing the cell index and the corresponding spot index. Output of get_cell_spot_pair().
        spots_to_cells_df: DataFrame with shape (n_spots, 2) containing the spot index and the corresponding cell index. Output of get_spot_cell_pair().

    Returns:
        The fraction of spots that match the reverse mapping.
    """
    # Count number of matches
    n_matches = 0
    for _, row in spots_to_cells_df.iterrows():
        spot_idx = row['spot index']  # current spot index
        cell_idx = row['cell index']  # matched cell index

        # Get spot matched spot index from cells_to_spots_df
        spot_idx_matched = cells_to_spots_df.loc[cells_to_spots_df['cell index'] == cell_idx, 'spot index'].values[0]

        # Check if they match
        if spot_idx == spot_idx_matched:
            n_matches += 1

    # Compute fraction of spots that match the reverse mapping
    matches_fraction = n_matches / len(spots_to_cells_df)

    print(f">>> Fraction of spots that match the reverse mapping: {matches_fraction:.2f}")

    return matches_fraction


# Compute the quality of deterministic mapping in two ways:
#         1. For non annotated spatial data: computes gene expression similarity between deterministic cell-spot pair.
#         2. For annotated spatial data: computes cell type (annotation) accuracy between deterministic cell-spot pair.
# Each is conditioned on filtering.

def deterministic_mapping_similarity(adata_map, adata_sc, adata_st, flavour: str, filter=False, threshold=0.5, plot=True):
    """
    Compute cosine similarity between the expression profiles, limited to the shared genes, of the deterministic cell-spot pair.

    Args:
        adata_map: Mapping object output of map_cells_to_space
        adata_sc: input single cell data
        adata_st: input spatial data
        flavour (str): Either "spot_to_cell" or "cell_to_spot" for the mapping perspective.
        filter (bool): Whether cell filtering is active. Default: False.
        threshold (float): Threshold value for cell filtering.
        plot (bool): Whether to plot the histogram of mapped similarities. Default: True.

    Returns:
        mapped_similarity (array): shape = (n_spots/n_cells, ) depending on the mapping perspective. Cosine similarity of the shared
        genes expression profile between the deterministic cell-spot pair.
    """

    # Make all gene names to lower case
    adata_sc.var_names = adata_sc.var_names.str.lower()
    adata_st.var_names = adata_st.var_names.str.lower()

    # Get deterministic cell-spot pair, always (spot_idx, cell_idx)
    if flavour == "spot_to_cell":
        # Each spot assigned to a cell
        pairs_df = get_spot_cell_pair(adata_map, filter=filter, threshold=threshold)
    elif flavour == "cell_to_spot":
        # Each cell assigned to a spot
        pairs_df = get_cell_spot_pair(adata_map, filter=filter, threshold=threshold)
    else:
        raise ValueError("Invalid flavour. Choose either 'spot_to_cell' or 'cell_to_spot'.")

    # Get shared genes
    shared_genes = adata_map.uns['training_genes']

    # Convert indices/names to a form AnnData accepts
    cell_idx = pairs_df['cell index']
    spot_idx = pairs_df['spot index']

    # If they are not integer dtype, convert to list (AnnData accepts obs_names as list)
    if cell_idx.dtype.kind not in {'i', 'u'}:
        cell_idx = cell_idx.tolist()
    if spot_idx.dtype.kind not in {'i', 'u'}:
        spot_idx = spot_idx.tolist()

    # Extract expression profiles using AnnData indexing (safe with names or integers)
    sc_profile = adata_sc[cell_idx, shared_genes].X
    st_profile = adata_st[spot_idx, shared_genes].X

    # Turn to array if sparse
    if hasattr(sc_profile, "toarray"):
        sc_profile = sc_profile.toarray()
    if hasattr(st_profile, "toarray"):
        st_profile = st_profile.toarray()

    # Ensure 2D arrays
    sc_profile = np.atleast_2d(np.asarray(sc_profile, dtype=float))
    st_profile = np.atleast_2d(np.asarray(st_profile, dtype=float))

    # Compute cosine similarity (no torch)
    # Numerator: row-wise dot product as element-wise multiplication and sum over rows
    num = np.sum(sc_profile * st_profile, axis=1)  # shape (n_filtered_cells/n_spots,)
    # Denominator: product of row-wise L2 norms
    denom = np.linalg.norm(sc_profile, axis=1) * np.linalg.norm(st_profile, axis=1)
    # Cossim
    mapped_similarity = num / denom

    if plot:
        # Histogram of cosine similarities
        plt.hist(mapped_similarity, bins=np.linspace(-1, 1, 50), edgecolor="black")
        plt.title("Histogram of pair similarities")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Number of pairs")
        plt.show()

    return mapped_similarity

# NOTE: Tangram's main use case is to project single cell annotation onto the spatial spots. This is done by assigning to each spot the
# cell type of the cell with the highest probability over it. A common way of evaluating this use is to compute the accuracy of
# cell type predictions wrt to the spatial ground truth which is either the result of biologically supervised annotation, clustering or
# synthetically derived.



### ANNOTATION ###
def deterministic_annotation(
        adata_map,
        adata_sc,
        adata_st,
        flavour: str,
        sc_cluster_label=None,
        st_cluster_label=None,
        filter=False,
        threshold=0.5,
        ):
    """
    Compute annotation transfer based on deterministic one-to-one mapping.
    Requires annotated spatial data, annotations must be coherent and harmonized for proper prediction.
    NOTE: With "cell_to_spot" flavoured mapping some spots might not be called and therefore annotated. To properly evaluate spatial
    annotation transfer use only "spot_to_cell". Flavour "cell_to_spot" is jet for legacy reasons.

    Args:
        adata_map: Mapping object output of map_cells_to_space.
        adata_sc: input single cell data.
        adata_st: input spatial data.
        flavour (str): Either "spot_to_cell" or "cell_to_spot" for the mapping perspective.
        sc_cluster_label: column name of single cell cluster/annotation labels.
        st_cluster_label: column name of spatial cluster/annotation labels.
        filter (bool): Whether cell filtering is active. Default: False.
        threshold (float): Threshold value for cell filtering.

    Returns:
        array-like objects with true and predicted annotations.
    """

    # Input
    if sc_cluster_label is None or st_cluster_label is None:
        raise ValueError("Provide cluster/annotation labels for single cell and spatial data.")
    if sc_cluster_label not in adata_sc.obs.columns or st_cluster_label not in adata_st.obs.columns:
        raise ValueError("Invalid cluster/annotation labels.")
    # Check mismatch in labels
    if len(set(adata_st.obs[st_cluster_label]) & set(adata_sc.obs[sc_cluster_label])) == 0:
        raise ValueError("No common labels between single cell and spatial data.")
    if not set(adata_st.obs[st_cluster_label].unique()).issubset(set(adata_sc.obs[sc_cluster_label].unique())):
        logging.warning('Annotation labels are not harmonized')

    # Make all gene names to lower case
    adata_sc.var_names = adata_sc.var_names.str.lower()
    adata_st.var_names = adata_st.var_names.str.lower()

    # Get deterministic spot-cell pair, always (spot_idx, cell_idx)
    if flavour == "spot_to_cell":
        # Each spot assigned to a cell
        pairs_df = get_spot_cell_pair(adata_map, filter=filter, threshold=threshold)
    elif flavour == "cell_to_spot":
        # Each cell assigned to a spot
        pairs_df = get_cell_spot_pair(adata_map, filter=filter, threshold=threshold)
    else:
        raise ValueError("Invalid flavour. Choose either 'spot_to_cell' or 'cell_to_spot'.")

    # Get true-predicted annotation pairs of shape (n_spots,) (originally pandas series)
    true_annotation = adata_st.obs[st_cluster_label].loc[pairs_df['spot index']].tolist()  # from st data
    pred_annotation = adata_sc.obs[sc_cluster_label].loc[pairs_df['cell index']].tolist()  # from sc data
    # NOTE: Both loc[...] calls return values in the order given by the corresponding column of pairs_df

    return true_annotation, pred_annotation

def transfer_annotation(adata_map, adata_st, sc_cluster_label, filter=False, threshold=0.5):
    """
    Transfer cell type (annotation) from single cell data to spatial data.
    Overwrites project_cell_annotation() and cell_type_mapping() of original tangram repo.

    Args:
        adata_map (AnnData): output of map_cells_to_space
        adata_st (AnnData): spatial data
        sc_cluster_label (str): column name of single cell cluster/annotation labels
        filter (bool): Whether cell filtering is active. Default: False.
        threshold (float): Threshold value for cell filtering.

    Returns:
        None.
        Update spatial Anndata by creating:
        1. `obsm` `tangram_` field with a dataframe with spatial prediction for each annotation (number_spots, number_annotations)
        2. 'obs' `tangram_annotation` field with the annotation with the highest probability over each spot (number_spots,)
    """
    # Controls
    if filter and "F_out" not in adata_map.obs.keys():
        raise ValueError("Missing final filter in mapped input data with filter=True.")
    if sc_cluster_label not in adata_map.obs.columns:
        raise ValueError("Invalid single cell data cluster/annotation labels.")

    # OHE single cell annotations (use tangram.utils function)
    df = ut.one_hot_encoding(adata_map.obs[sc_cluster_label])

    # Annotations probabilities dataframe (bulk-vote transfer) A_st = M^T @ A_sc
    if filter:
        # Restrict to filtered cells
        df_ct_prob = adata_map[adata_map.obs["F_out"] > threshold].X.T @ df.loc[adata_map.obs["F_out"] > threshold]
    else:
        df_ct_prob = adata_map.X.T @ df

    # Assign spot indexes
    df_ct_prob.index = adata_st.obs.index

    # Normalize per cell type (max-min)
    vmin = df_ct_prob.min()
    vmax = df_ct_prob.max()
    df_ct_prob = (df_ct_prob - vmin) / (vmax - vmin)

    # Add fields to spatial AnnData
    adata_st.obsm["tangram_cluster_probs"] = df_ct_prob  # (number_spots, number_annotations)
    adata_st.obs["tangram_annotation"] = df_ct_prob.idxmax(axis=1)  # (number_spots,)


def annotation_report(true_annotation, pred_annotation):
    """
    Generate a full classification report and overall accuracy
    between ground-truth and predicted annotations.

    Args:
        true_annotation : array-like of shape (n_samples,)
        pred_annotation : array-like of shape (n_samples,)

    Returns:
        annotation_acc : float
            Overall accuracy
        report_df : pd.DataFrame
            Per-class metrics (precision, recall, f1, support)
            plus macro/weighted averages.
    """
    # Overall accuracy
    annotation_acc = accuracy_score(true_annotation, pred_annotation)

    # Per-class metrics (precision, recall, f1, support)
    report_dict = classification_report(
        true_annotation,
        pred_annotation,
        output_dict=True,
        zero_division=0  # avoid division errors for empty classes
    )

    # Convert to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Optional: round values for neat printing
    report_df = report_df.round(3)

    # Display
    print(f"\nOverall annotation accuracy: {annotation_acc:.3f}")
    print("\nPer-class annotation report:")
    print(report_df.to_string())

    return annotation_acc, report_df

## FIlTER EVALUATION ##

def filter_cell_choice_consistency(filter_square, threshold=0.5):
    """
    Evaluate cell choice consistency across multiple runs.

    Arg:
        filter_square (array shape = (n_cells, n_runs)): Square containing in each column the final filter values of each run.
        threshold (float): Threshold value for cell filtering.
    Returns:
        Prints a histogram of the number of runs each cell is selected in (is above threshold).
        Reports the filter consistency as the percentage, over all cells, of cells that are selected in all runs.
    """

    # Get number of runs
    n_runs = filter_square.shape[1]

    # Count number of runs each cell is selected in
    selected_cells = np.sum(filter_square > threshold, axis=1)

    # Plot histogram of selected cells
    plt.hist(selected_cells, bins=np.arange(0, n_runs+1), edgecolor="black")
    plt.title("Histogram of selected cells")
    plt.xlabel("Number of selected runs")
    plt.show()

    # Compute consistency
    consistency = np.sum(selected_cells == n_runs) / len(selected_cells)

    print(f"Filter consistency: {consistency * 100:.3f} %")

    return consistency

# TODO: as of now filter_value > threshold, alternatively move everything to filter_value >= threshold