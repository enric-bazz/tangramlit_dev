"""
This module implements spatial binning of spatial transcriptimocs data with a Farthest-Point Sampling (FPS) seed selection and Breadth-First Search (BFS) bin growth algorithm. 
It also computes cell type fractions within each spatial bin.
Its intended use is to create spatial bins of approximately equal size (number of cells) that are spatially contiguous, which can be used for conditional priors in Tangram.
The conditional priors can be defined as the relative frequencies of cell types (e.g. clusters) within each spatial bin, 
which can help guide Tangram's mapping of single cells to spatial locations by providing local cell type composition information (ST annotation supervision).
"""

import pandas as pd
import numpy as np
from collections import deque
from sklearn.metrics import pairwise_distances


def fps_seeds(coords, num_seeds, random_state=None):
    """
    Farthest-Point Sampling (FPS) to select well-spread seeds.
    coords: (n_cells,2) array of spatial positions
    num_seeds: number of seeds to select
    returns: list of seed indices
    """
    n = coords.shape[0]
    if random_state is not None:
        np.random.seed(random_state)
    seeds = [np.random.randint(n)]
    min_dists = pairwise_distances(coords.values, coords.iloc[seeds].values).flatten()

    for _ in range(1, num_seeds):
        idx = np.argmax(min_dists)
        seeds.append(idx)
        dist_to_new = np.linalg.norm(coords.values - coords.iloc[idx].values, axis=1)
        min_dists = np.minimum(min_dists, dist_to_new)

    return seeds


def bfs_bin_growth(coords_df, seeds, adj_matrix, target_size=50):
    """
        Grow bins from seeds using BFS on a sparse adjacency matrix.
        coords: (n_cells,2)
        seeds: list of seed indices
        adj_matrix: sparse CSR (n_cells x n_cells), e.g., Squidpy 6-NN connectivities
        target_size: approximate number of cells per bin
        returns: labels array (n_cells,) bin assignment
    """

    n = coords_df.shape[0]
    labels = -np.ones(n, dtype=int)
    bin_id = 0
    adj = adj_matrix.tocsr()

    for s in seeds:
        if labels[s] != -1:
            continue
        q = deque([s])
        labels[s] = bin_id
        count = 1
        while q and count < target_size:
            v = q.popleft()
            neighbors = adj.indices[adj.indptr[v]:adj.indptr[v + 1]]
            for nb in neighbors:
                if labels[nb] == -1:
                    labels[nb] = bin_id
                    q.append(nb)
                    count += 1
                    if count >= target_size:
                        break
        bin_id += 1

    # Assign leftover cells
    leftover = np.where(labels == -1)[0]
    if leftover.size > 0:
        bin_ids = np.unique(labels[labels >= 0])
        centroids = np.vstack([coords_df.iloc[labels == b].values.mean(axis=0) for b in bin_ids])
        from sklearn.neighbors import NearestNeighbors
        tree = NearestNeighbors(n_neighbors=1).fit(centroids)
        _, nearest = tree.kneighbors(coords_df.iloc[leftover].values)
        labels[leftover] = bin_ids[nearest[:, 0]]

    return labels


def compute_bin_celltype_fractions(adata, cluster_label='cluster', bin_label='bin'):
    """
    Compute relative frequencies of cell types in each spatial bin.

    Parameters
    ----------
    adata : AnnData
        AnnData object with obs[cluster_label] and obs[bin_label].
    cluster_label : str
        Column in adata.obs indicating cell type or cluster.
    bin_label : str
        Column in adata.obs indicating bin assignment.

    Returns
    -------
    pd.DataFrame
        Rows: cell types, Columns: bins, values: fraction of cells of that type in the bin.
    """
    # Crosstab counts: cell types vs bins
    counts = pd.crosstab(adata.obs[cluster_label], adata.obs[bin_label])

    # Convert counts to fractions per bin
    fractions = counts.divide(counts.sum(axis=0), axis=1)

    return fractions