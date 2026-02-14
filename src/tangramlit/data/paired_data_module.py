import logging
import numpy as np
import scanpy as sc
import squidpy as sq
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .paired_dataset import AdataPairDataset

### DataModule Class ###

class PairedDataModule(LightningDataModule):
    """
    Lightning DataModule for Tangram mapping.
    """

    def __init__(self,
                 adata_sc=None,
                 adata_st=None,
                 input_genes=None,
                 train_genes_names=None,
                 val_genes_names=None,
                 cluster_label=None,
                 ):
        """
        Preprocess single-cell and spatial anndata objects.

        Args:
            adata_sc (AnnData): Single-cell AnnData object.
            adata_st (AnnData): Spatial AnnData object.
            input_genes (list): List of input genes to use for training. If None, use all genes shared between adata_sc and adata_st.
            train_genes_names (list): List of names of genes to use for training. If None, use all genes shared between adata_sc and adata_st.
            val_genes_names (list): List of names of genes to use for validation.
            cluster_label (str): Field in `adata_sc.obs` used for clustering/annotating single cell data.
        """
        super().__init__()
        self.adata_sc = adata_sc
        self.adata_st = adata_st
        self.input_genes = input_genes  # Allow passing specific genes for training
        self.train_genes_names = train_genes_names
        self.val_genes_names = val_genes_names
        self.cluster_label = cluster_label

        # Turn all input gene names to lowercase
        if self.input_genes is not None:
            self.input_genes = [g.lower() for g in self.input_genes]
        if self.train_genes_names is not None:
            self.train_genes_names = [g.lower() for g in self.train_genes_names]
        if self.val_genes_names is not None:
            self.val_genes_names = [g.lower() for g in self.val_genes_names]

        # Compute spatial neighbors needed for the neighborhood extension of Tangram
        if not all(k in adata_st.obsp.keys() for k in ['spatial_connectivities', 'spatial_distances']):
            sq.gr.spatial_neighbors(self.adata_st, set_diag=False, key_added="spatial")


    def prepare_data(self):
        """
        Takes anndata objects and prepares them for mapping. It does not slice/subset the datasets on the training genes.
        By default, the training genes set is made of the shared genes that are expressed in at least one obs in both datasets.
        Executed before setup() in the trainer.fit call.
        """
        # Preprocess data - define adata.uns['training_genes'] - originally implemented in tg.mapping_utils.pp_adatas()
        print("Preprocessing data...")

        # 1. Put all var indexes to lower case to align
        self.adata_sc.var.index = [g.lower() for g in self.adata_sc.var.index]
        self.adata_st.var.index = [g.lower() for g in self.adata_st.var.index]

        # 2. Make genes unique
        self.adata_sc.var_names_make_unique()
        self.adata_st.var_names_make_unique()

        # 3. Annotate sparsity on all genes
        annotate_gene_sparsity(self.adata_sc)
        annotate_gene_sparsity(self.adata_st)

        # 4. Define shared genes set and annotate in adata.uns['overlap_genes']
        overlap_genes = set(self.adata_sc.var.index) & set(self.adata_st.var.index)
        self.adata_sc.uns['overlap_genes'] = list(overlap_genes)
        self.adata_st.uns['overlap_genes'] = list(overlap_genes)

        # 5. Filter all-zero-valued genes and get filtered overlap (candidate training genes)
        # scanpy.pp.filter_genes() returns tuple containing: (a boolean array indicating which genes were filtered out, the number of cells per gene)
        filtered_genes_sc, _ = sc.pp.filter_genes(self.adata_sc, min_cells=1, inplace=False)  # skip n_cells output
        filtered_genes_st, _ = sc.pp.filter_genes(self.adata_st, min_cells=1, inplace=False)  # skip n_cells output
        # Get gene names by masking the adata with boolean arrays
        filtered_genes_sc = self.adata_sc[:, filtered_genes_sc].var_names
        filtered_genes_st = self.adata_st[:, filtered_genes_st].var_names
        overlap_filtered_genes = set(filtered_genes_sc) & set(filtered_genes_st)

        # 6. Define training genes as intersection of input training genes and overlapping filtered genes
        if self.input_genes is not None:
            training_genes = list(set(self.input_genes) & overlap_filtered_genes)
            print(f"Using {len(training_genes)} training genes provided by user.")
        else:
            training_genes = list(overlap_filtered_genes)
        print(f"Using {len(training_genes)} shared marker genes.")

        # 7. Annotate training genes set in adata.uns['training_genes'] and adata.var['is_training']
        self.adata_sc.uns["training_genes"] = training_genes
        self.adata_st.uns["training_genes"] = training_genes
        if self.train_genes_names is None:
            self.adata_sc.var["is_training"] = self.adata_sc.var.index.isin(training_genes)
            self.adata_st.var["is_training"] = self.adata_st.var.index.isin(training_genes)
        else:
            mask_sc = self.adata_sc.var.index.isin(training_genes) & self.adata_sc.var.index.isin(self.train_genes_names)
            mask_st = self.adata_st.var.index.isin(training_genes) & self.adata_st.var.index.isin(self.train_genes_names)
            self.adata_sc.var["is_training"] = mask_sc
            self.adata_st.var["is_training"] = mask_st
        # NOTE: adata.var['is_training'] == True iff the gene is both in the filtered shared genes (trainig_genes) and,
        # if applicable, also selected in train_genes_names
        if self.val_genes_names is not None:
            mask_sc = self.adata_sc.var.index.isin(training_genes) & self.adata_sc.var.index.isin(self.val_genes_names)
            mask_st = self.adata_st.var.index.isin(training_genes) & self.adata_st.var.index.isin(self.val_genes_names)
            self.adata_sc.var["is_validation"] = mask_sc
            self.adata_st.var["is_validation"] = mask_st


    def setup(self, stage: str):
        """
        Set up datasets for use in dataloaders.
        This method is called on every GPU separately.
        Execute after prepare_data() and before train/val_dataloader().
        """
        self.train_dataset = AdataPairDataset(self.adata_sc,
                                              self.adata_st,
                                              genes_names=self.train_genes_names,
                                              cluster_label=self.cluster_label,
                                              train=True,
                                              )
        self.val_dataset = AdataPairDataset(self.adata_sc,
                                            self.adata_st,
                                            genes_names=self.val_genes_names,
                                            train=False,
                                            )
        # Warning message if no validation genes are provided
        if self.val_genes_names is None:
            logging.warning("No validation genes specified. Using all genes for validation.")


    def train_dataloader(self):
        """
        Return a DataLoader for training.
        For Tangram, we use a single batch containing all data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # always use batch_size=1 as each item contains all data
            shuffle=False,  # no need to shuffle as we have just one batch
            num_workers=0,  # process in the main thread
            pin_memory=True,  # speed up data transfer to GPU if using CUDA
            collate_fn=lambda x: x[0]  # prevent adding batch dimension [1, n_cells/spots, n_genes] -> [n_cells/spots, n_genes]
        )

    def val_dataloader(self):
        """
            Return a DataLoader for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # always use batch_size=1 as each item contains all data
            shuffle=False,  # no need to shuffle as we have just one batch
            num_workers=0,  # process in the main thread
            pin_memory=True,  # speed up data transfer to GPU if using CUDA
            collate_fn=lambda x: x[0]  # prevent adding batch dimension [1, n_cells/spots, n_genes] => [n_cells/spots, n_genes]
        )



### Utility functions ###

def annotate_gene_sparsity(adata: sc.AnnData):
    """
    Annotates gene sparsity in the given AnnData object.
    Update the given anndata by creating a `var` "sparsity" field with gene_sparsity (1 - % non-zero observations).

    Args:
        adata (AnnData): single cell or spatial data.

    Returns:
        None
    """
    sparsity_arr = (adata.X != 0).mean(axis=0) 
    adata.var["sparsity"] = 1 - (sparsity_arr.A1 if hasattr(sparsity_arr, "A1") else np.ravel(sparsity_arr))  # .A1 flattens sparse matrix to 1D dense array