"""
Test script to validate spatial binning to generate cell type fraction priors for Tangram.
Note that this extension requires annotation of the spatial data with cell type labels, and is not yet integrated into the main Tangram pipeline. 
It is intended for testing and demonstration purposes.
"""

import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import numpy as np

from tangramlit.data.spatial_binning import bfs_bin_growth, fps_seeds, compute_bin_celltype_fractions


def main():
    adata_st = sc.read("C:/Users/enric/desktop/tangram_repo_dump/myDataCropped/slice200_norm_reduced.h5ad")
    print(adata_st)

    sq.gr.spatial_neighbors(adata_st, set_diag=False, key_added="spatial")
    print(adata_st.obsp.keys())

    coords = adata_st.obsm['spatial']
    adj_matrix = adata_st.obsp['spatial_connectivities']  # Squidpy 6-NN graph
    n_cells = coords.shape[0]
    target_size = 25
    num_bins = int(np.ceil(n_cells / target_size))

    # 1. Select well-spread seeds
    seeds = fps_seeds(coords, num_seeds=num_bins, random_state=0)

    # 2. Grow bins
    labels = bfs_bin_growth(coords, seeds, adj_matrix, target_size=target_size)

    # 3. Save to AnnData
    adata_st.obs['bin'] = labels
    adata_st.obs['bin_cell_count'] = adata_st.obs.groupby('bin')['bin'].transform('count')

    # 4. Plot
    plt.figure(figsize=(6,6))
    scatter = plt.scatter(coords.values[:,0], coords.values[:,1], c=labels, cmap='tab20', s=20)  # retrieve df values
    plt.gca().invert_yaxis()  # optional: match typical tissue orientation
    plt.axis('equal')
    plt.xlabel('X (µm)')
    plt.ylabel('Y (µm)')
    plt.title('Spatial bins')
    plt.colorbar(scatter, label='Bin ID')
    plt.show()


    ## Conditional prior
    cluster_label = 'class_label'  # or 'sublclass'

    fractions_df = compute_bin_celltype_fractions(adata_st, cluster_label=cluster_label, bin_label='bin')
    fractions_df.head()

    # visualize
    coords = adata_st.obsm['spatial'].values
    bins = adata_st.obs['bin'].values
    clusters = adata_st.obs[cluster_label].values

    # Map clusters to a small set of markers
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']
    cluster_to_marker = {c: markers[i % len(markers)] for i, c in enumerate(np.unique(clusters))}

    plt.figure(figsize=(6,6))
    for c in np.unique(clusters):
        idx = clusters == c
        plt.scatter(coords[idx,0], coords[idx,1], c=bins[idx], cmap='tab20', s=20,
                    marker=cluster_to_marker[c], label=str(c))
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.legend(markerscale=1, fontsize=8, title='Cluster')
    plt.show()

if __name__ == "__main__":
    main()