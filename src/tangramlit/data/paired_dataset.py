import logging
import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import torch
from scipy.sparse import csc_matrix, csr_matrix
from torch.utils.data import Dataset

#### Dataset Class ###

class AdataPairDataset(Dataset):
    """
    Dataset class for single-cell and spatial anndata objects.
    Returns a single batch containing all data, sliced according to the provided names and based on the current mode.

    Args:
        adata_sc (AnnData): Single-cell AnnData object.
        adata_st (AnnData): Spatial AnnData object.
        genes_names (list): List of gene names to use for training/testing depending on the call.
            If None, use all genes shared between adata_sc and adata_st.
        cluster_label (str): Field in `adata_sc.obs` used for clustering/annotating single cell data.
    """
    def __init__(self,
                 adata_sc,
                 adata_st,
                 genes_names=None,
                 cluster_label=None,
                 train=None,
                 ):

        
        self.train=train
        # Get training genes from adata.uns['training_genes'] - defined in LightningDataModule.prepare_data()
        training_genes = adata_sc.uns['training_genes']

        # S matrix (single-cell)
        if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
            self.S = torch.tensor(adata_sc[:, training_genes].X.toarray(), dtype=torch.float32)
        elif isinstance(adata_sc.X, np.ndarray):
            self.S = torch.tensor(adata_sc[:, training_genes].X, dtype=torch.float32)
        else:
            X_type = type(adata_sc.X)
            logging.error(f"Single-cell AnnData X has unrecognized type: {X_type}")
            raise NotImplementedError

        # G matrix (spatial)
        if isinstance(adata_st.X, csc_matrix) or isinstance(adata_st.X, csr_matrix):
            self.G = torch.tensor(adata_st[:, training_genes].X.toarray(), dtype=torch.float32)
        elif isinstance(adata_st.X, np.ndarray):
            self.G = torch.tensor(adata_st[:, training_genes].X, dtype=torch.float32)
        else:
            X_type = type(adata_st.X)
            logging.error(f"Spatial AnnData X has unrecognized type: {X_type}")
            raise NotImplementedError
        
        if self.train:

            # Spatial Graph (both objects returned by squidpy.gr.spatial_neighbors() are of type csr_matrix)
            print('Allocating nearest neighbor graphs to dense tensors...')

            # Connectivities
            self.spatial_graph_conn = torch.tensor(
                adata_st.obsp['spatial_connectivities'].toarray(),
                dtype=torch.float32
            )
            conn_size_mib = self.spatial_graph_conn.element_size() * self.spatial_graph_conn.numel() / (1024 ** 2)
            print(
                f"spatial_graph_conn allocated: shape={tuple(self.spatial_graph_conn.shape)}, size={conn_size_mib:.2f} MiB"
            )

            # Distances
            self.spatial_graph_dist = torch.tensor(
                adata_st.obsp['spatial_distances'].toarray(),
                dtype=torch.float32
            )
            dist_size_mib = self.spatial_graph_dist.element_size() * self.spatial_graph_dist.numel() / (1024 ** 2)
            print(
                f"spatial_graph_dist allocated: shape={tuple(self.spatial_graph_dist.shape)}, size={dist_size_mib:.2f} MiB"
            )

            print('Done.')

            # A matrix (cluster/annotation OHE)
            if cluster_label is not None:  # run only on self.train_dataset() call
                self.A = ohe_cluster_label(adata_sc.obs[cluster_label])

        # Get train/val genes indexes from names
        self.genes_idx = gene_names_to_indices(gene_names=genes_names, adata=adata_st) if genes_names is not None else slice(None)
        # NOTE: When indices are `None`, it defaults to using all genes for both training and validation
        # Since this behavior might be undesirable, a warning message is displayed after the AdataPairDataset() call in setup()

        # Store metadata
        self.training_genes = training_genes
        self.n_cells = self.S.shape[0]
        self.n_spots = self.G.shape[0]
        self.n_genes = len(training_genes)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Returns sliced S and G tensors according to input indexes (for training or validation).
        Includes spatial graphs and A only during training.
        """
        batch = {
            'S': self.S[:, self.genes_idx],
            'G': self.G[:, self.genes_idx],
            'genes_number': len(self.genes_idx) if not isinstance(self.genes_idx, slice) else self.n_genes,
        }

        if self.train:
            batch['spatial_graph_conn'] = self.spatial_graph_conn
            batch['spatial_graph_dist'] = self.spatial_graph_dist
            if hasattr(self, "A"):
                batch['A'] = self.A
            

        return batch



### Utility functions ###

def gene_names_to_indices(gene_names, adata):
    """
        Get indices of genes in AnnData object's var, handling case sensitivity.
        Only includes genes that are present in adata.uns['training_genes'].

        Args:
            gene_names (list): List of gene names to find indices for
            adata (AnnData): AnnData object to search in

        Returns:
            list: List of indices corresponding to the input gene names

        Raises:
            ValueError: If any gene name is not found in the AnnData object
            KeyError: If 'training_genes' is not present in adata.uns
    """

    # Find indices for each gene name
    indices = []
    missing_genes = []

    for gene in gene_names:
        gene_lower = gene.lower()
        # Check if gene is in training_genes
        if gene_lower in adata.uns['training_genes']:
            indices.append(adata.uns['training_genes'].index(gene_lower))
        else:
            missing_genes.append(gene)  # use .item() if array

    if missing_genes:
        logging.warning(f"The following train/val input genes were removed with preprocessing: {missing_genes}.")

    return indices



def ohe_cluster_label(cluster_labels):
    """
        Given a Series of cluster labels, returns a one-hot encoded tensor.

        Args:
            cluster_labels (pandas.Series): labels to OHE.

        Returns:
            torch.Tensor: shape (n_cells, n_clusters), dtype=float32.
    """
    labels = pd.Categorical(cluster_labels)
    label_codes = torch.tensor(labels.codes, dtype=torch.long)  # torch.nn.functional.one_hot() accepts torch.long
    A = torch.nn.functional.one_hot(label_codes, num_classes=len(labels.categories)).float()
    
    return A