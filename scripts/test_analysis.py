"""
Test script for tangramlit mapping and analysis public API.
"""

import yaml
import anndata as ad
import tangramlit as tgl

def load_data():
    # read adata
    adata_sc = ad.read_h5ad("C:/Users/enric/tangramlit_dev/data/test_sc.h5ad")
    adata_st = ad.read_h5ad("C:/Users/enric/tangramlit_dev/data/test_slice200.h5ad")

    # read config
    with open(f"C:/Users/enric/tangramlit_dev/data/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return (adata_sc, adata_st), config

def train_mapper(adata_sc, adata_st, config):
    # Train/val split
    train_genes, val_genes, _ = tgl.split_train_val_test_genes(adata_sc, adata_st)

    # Filter check
    if config['filter'] and (config['target_count'] == 'None'):
        config['target_count'] = None  # remove string formatting

    # Training
    adata_map, mapper, datamodule = tgl.map_cells_to_space(
        adata_sc=adata_sc,
        adata_st=adata_st,
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        **config
        )
    
    return adata_map, mapper, datamodule

def validate_mapper(adata_map, mapper, datamodule):
    # Validate
    results = tgl.validate_mapping_experiment(mapper, datamodule)

    # sc gene projection
    ad_ge = tgl.project_sc_genes_onto_space(adata_map, datamodule)
    df = tgl.compare_spatial_geneexp(ad_ge, datamodule)
    tgl.plot_training_scores(df)
    tgl.plot_auc_curve(df, plot_train=True, plot_validation=True)

    tgl.plot_validation_metrics_history(adata_map=adata_map, add_training_scores=True)

    return results

def test_analysis(adata_sc, adata_st, adata_map, mapper, datamodule, config):
    # sc gene projection
    ad_ge = tgl.project_sc_genes_onto_space(adata_map, datamodule)
    df = tgl.compare_spatial_geneexp(ad_ge, datamodule)
    tgl.plot_training_scores(df)
    tgl.plot_auc_curve(df, plot_train=True, plot_validation=True)  # works only with validation genes

    # Plot loss terms
    tgl.plot_training_history(adata_map=adata_map, hyperpams=mapper.hparams, log_scale=False, lambda_scale=True)
    
    tgl.plot_loss_terms(adata_map=adata_map, loss_key=['count_reg', 'main_loss', 'ct_island_term'], lambda_coeff=[1.0e-3, 1, 1])
    tgl.plot_loss_terms(adata_map=adata_map, loss_key=['count_reg', 'main_loss', 'ct_island_term'], lambda_coeff=[1.0e-3, 1, 1],
                                                       lambda_scale=True, make_subplot=True, subplot_shape=(1,3))
    
    tgl.plot_validation_metrics_history(adata_map=adata_map, add_training_scores=True)

    # Plot final filter values distribution
    if config["filter"]:
        tgl.plot_filter_weights(adata_map, plot_heatmap=False, plot_spaghetti=True, plot_envelope=True)
        tgl.plot_filter_count(adata_map, target_count=config['target_count'])
        filt_corr = tgl.compute_filter_corr(adata_map, plot=True)

    # Try analysis functions
    df1 = tgl.get_cell_spot_pair(adata_map, filter=True)
    df2 = tgl.get_spot_cell_pair(adata_map, filter=True)


    # Shared labels
    labels = set(adata_st.obs['class_label']) & set(adata_sc.obs['class_label'])
    # Accuracy
    true, pred = tgl.deterministic_annotation(adata_map, adata_sc, adata_st,
                                sc_cluster_label='class_label', st_cluster_label='class_label',
                                flavour='spot_to_cell', filter=False)
    acc, stats = tgl.annotation_report(true, pred)

    map_sim = tgl.deterministic_mapping_similarity(adata_map, adata_sc, adata_st, flavour='spot_to_cell', filter=config['filter'])


    # Annotation transfer test
    tgl.transfer_annotation(adata_map=adata_map, adata_st=adata_st, sc_cluster_label='class_label',
                        filter=config['filter'])
    
    # Benchmarking
    df_g = tgl.benchmark_mapping(ad_ge, datamodule, df)
    print(df_g.head())


def main():

    (adata_sc, adata_st), config = load_data()

    adata_map, mapper, datamodule = train_mapper(adata_sc=adata_sc, adata_st=adata_st, config=config)

    # Validate the mapper
    validate_mapper(adata_map=adata_map, mapper=mapper, datamodule=datamodule)

    # Test analysis
    test_analysis(adata_sc=adata_sc, adata_st=adata_st, adata_map=adata_map, mapper=mapper, datamodule=datamodule, config=config)


if __name__ == "__main__":
    main()



    #TODO: create a script that takes the two adata as input and a trainin config and 
    # writes in output three anndata: the two input preprocessed and the adata_map.

    #TODO: create a script that performs an optuna tuning of lambdas and automiatically
    # trains a model to completion with that configuration.