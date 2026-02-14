"""
Script to cross-validate genes with Tangram mapping.
"""

import anndata as ad
import yaml

from tangramlit.mapping.trainer import map_cells_to_space
from tangramlit.mapping.utils import validate_mapping_experiment
from tangramlit.data.genes_splits import kfold_gene_splits

def cross_validate_mapping(
        adata_sc,
        adata_st,
        k=5,
        input_genes=None,
        train_genes_names=None,
        val_genes_names=None,
        filter=False,
        learning_rate=0.1,
        num_epochs=1000,
        random_state=None,
        lambda_d=0,
        lambda_g1=1,
        lambda_g2=0,
        lambda_r=0,
        lambda_l1=0,
        lambda_l2=0,
        lambda_count=1,
        lambda_f_reg=1,
        target_count=None,
        lambda_sparsity_g1=0,
        lambda_neighborhood_g1=0,
        lambda_getis_ord=0,
        lambda_moran=0,
        lambda_geary=0,
        lambda_ct_islands=0,
        cluster_label=None,
        experiment_name="",
):
    """
    Executes genes set cross-validation using Lightning mapper.
    The genes set must be a subset of the shared genes set: verified with the validate_mapping_inputs() call inside map_cells_to_space().

    Args:
        adata_sc (AnnData): single cell data
        adata_st(AnnData): gene spatial data
        mode (str): Optional. Tangram mapping mode. Currently supported: 'vanilla', 'filter', 'refined'. Default is 'vanilla'.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        k (int): Number of folds for k-folds cross-validation. Default is 10.
        input_genes (list): Optional. Set of genes to be used for cross-validation.
        lambda_d (float): Optional. Strength of density regularizer. Default is 0.
        lambda_g1 (float): Optional. Strength of Tangram loss function. Default is 1.
        lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        lambda_sparsity_g1 (float): Optional. Strength of sparsity regularizer. Default is 0. Only valid when mode == 'refined'.
        lambda_neighborhood_g1 (float): Optional. Strength of neighborhood regularizer. Default is 0. Only valid when mode == 'refined'.
        lambda_getis_ord (float): Optional. Strength of Getis-Ord regularizer. Default is 0. Only valid when mode == 'refined'.
        lambda_moran (float): Optional. Strength of Moran regularizer. Default is 0. Only valid when mode == 'refined'.
        lambda_geary (float): Optional. Strength of Geary regularizer. Default is 0. Only valid when mode == 'refined'.
        lambda_ct_islands (float): Optional. Strength of ct islands enforcement. Default is 0. Only valid when mode == 'refined'.
        cluster_label (str): Optional. Field in `adata_sc.obs` used for aggregating single cell data. Only valid for `mode=refined` and lambda_ct-islands > 0. Default is None.
        random_state (int): Optional. pass an int to reproduce training. Default is None.

    Returns:
        cv_metrics (dict): Dictionary containing average metric scores across all folds
    """


    curr_cv_set = 1

    # Init fold metrics dictionary
    cv_metrics = {}

    # brign gene names to lowercase
    adata_sc.var_names = adata_sc.var_names.str.lower()
    adata_st.var_names = adata_st.var_names.str.lower()

    input_genes = input_genes if input_genes is not None else sorted(set(adata_sc.var_names) & set(adata_st.var_names))
    for train_genes, test_genes in kfold_gene_splits(genes=input_genes, k=5, random_state=random_state):

        print(f"Enter loop with test_genes:{test_genes}")
        # Train mapper
        adata_map, mapper, datamodule = map_cells_to_space(
            adata_sc=adata_sc,
            adata_st=adata_st,
            input_genes=input_genes,  # input genes list
            train_genes_names=train_genes,  # training genes of current fold
            val_genes_names=test_genes,  # validation genes of current fold
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            lambda_d=lambda_d,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_r=lambda_r,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_count=lambda_count,
            lambda_f_reg=lambda_f_reg,
            target_count=target_count,
            lambda_sparsity_g1=lambda_sparsity_g1,
            lambda_neighborhood_g1=lambda_neighborhood_g1,
            lambda_getis_ord=lambda_getis_ord,
            lambda_moran=lambda_moran,
            lambda_geary=lambda_geary,
            lambda_ct_islands=lambda_ct_islands,
            cluster_label=cluster_label,
            random_state=random_state,
        )

        # Compute current fold metrics
        fold_metrics = validate_mapping_experiment(model=mapper, datamodule=datamodule)[0]
        # Update cv dictionary
        for key, value in fold_metrics.items():
            if key not in cv_metrics:
                cv_metrics[key] = []  # first time, make a list
            cv_metrics[key].append(value)  # append new fold value


    # Calculate average metrics across folds
    """cv_metrics = {}
    for metric in metrics:
        temp_arr = np.zeros(len(fold_metrics[metric]))  # shape = (k,)
        for fold in range(len(fold_metrics[metric])):
            temp_arr[fold] = np.mean(fold_metrics[metric][fold])  # scalar
        cv_metrics[metric] = np.array(temp_arr, dtype='float32').mean().item()  # assing metric mean over folds"""

    return cv_metrics

def main():
    # gets input data and training config path and output folder path from user, then runs cross-validation and saves results in output folder
     # read adata
    adata_sc = ad.read_h5ad("C:/Users/enric/desktop/tangram_repo_dump/myDataCropped/test_sc_crop.h5ad")
    adata_st = ad.read_h5ad("C:/Users/enric/desktop/tangram_repo_dump/myDataCropped/slice200_norm_reduced.h5ad")

    # read config
    with open(f"C:/Users/enric/desktop/tangram_repo_dump/myDataCropped/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

        
    cv_results = cross_validate_mapping(adata_sc=adata_sc, adata_st=adata_st, k=5, **config)

    return cv_results


if __name__ == "__main__":
    main()