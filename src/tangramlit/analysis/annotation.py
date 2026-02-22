"""
Tangram's main use case is to project single cell annotation onto the spatial spots. This is done by assigning to each spot the
# cell type of the cell with the highest probability over it. A common way of evaluating this use is to compute the accuracy of
# cell type predictions wrt to the spatial ground truth which is either the result of biologically supervised annotation, clustering or
# synthetically derived.
"""

import logging
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from .deterministic_mapping import get_spot_cell_pair, get_cell_spot_pair

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
    df = one_hot_encoding(adata_map.obs[sc_cluster_label])

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


def one_hot_encoding(l, keep_aggregate=False):
    """
    Given a sequence, returns a DataFrame with a column for each unique value in the sequence and a one-hot-encoding.

    Args:
        l (sequence): List to be transformed.
        keep_aggregate (bool): Optional. If True, the output includes an additional column for the original list. Default is False.

    Returns:
        A DataFrame with a column for each unique value in the sequence and a one-hot-encoding, and an additional
        column with the input list if 'keep_aggregate' is True.
        The number of rows are equal to len(l).
    """
    df_enriched = pd.DataFrame({"cl": l})
    for i in l.unique():
        df_enriched[i] = list(map(int, df_enriched["cl"] == i))
    if not keep_aggregate:
        del df_enriched["cl"]
    return df_enriched