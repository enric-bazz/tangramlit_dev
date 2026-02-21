import anndata as ad
import numpy as np
import tangramlit as tgl
import yaml

def load_data(data_path, spatial_tech):
    # read adata
    adata_sc = ad.read_h5ad("C:/Users/enric/desktop/tangram_repo_dump/myDataCropped/test_sc_crop.h5ad")
    adata_st = ad.read_h5ad("C:/Users/enric/desktop/tangram_repo_dump/myDataCropped/slice200_norm_reduced.h5ad")

    # read config
    with open(f"C:/Users/enric/desktop/tangram_repo_dump/myDataCropped/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return (adata_sc, adata_st), config

def train_mapper(adata_sc, adata_st, config):
    # Train/val split
    # Get shared genes (case-insensitive)
    sc_genes = {gene.lower(): gene for gene in adata_sc.var_names}
    st_genes = {gene.lower(): gene for gene in adata_st.var_names}

    # Find intersection of lowercase gene names
    shared_genes_set = set(sc_genes.keys()) & set(st_genes.keys())
    shared_genes = [gene_lower for gene_lower in shared_genes_set]

    # Shuffle the shared genes
    shared_genes = np.array(shared_genes)
    if config['random_state'] is not None:
        np.random.seed(config['random_state'])
    np.random.shuffle(shared_genes)

    # Split into train and validation
    train_ratio = 0.8
    n_train = int(len(shared_genes) * train_ratio)
    train_genes = shared_genes[:n_train]
    val_genes = shared_genes[n_train:]

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
    #Validate
    results = tgl.validate_mapping_experiment(mapper, datamodule)

    # sc gene projection
    ad_ge = tgl.project_sc_genes_onto_space(adata_map, datamodule)
    df = tgl.compare_spatial_geneexp(ad_ge, datamodule)
    tgl.plot_training_scores(df)
    tgl.plot_auc_curve(df, plot_train=True, plot_validation=True)  # works only with validation genes

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

    dataset_number = 3

    # data_path = f"/nfsd/sysbiobig/bazzaccoen/tangramlit_dev/data/Dataset{dataset_number}/"

    (adata_sc, adata_st), config = load_data(data_path='blank', spatial_tech='seqFISH')

    adata_map, mapper, datamodule = train_mapper(adata_sc=adata_sc, adata_st=adata_st, config=config)

    # Validate the mapper
    results = validate_mapper(adata_map=adata_map, mapper=mapper, datamodule=datamodule)

    # Test analysis
    test_analysis(adata_sc=adata_sc, adata_st=adata_st, adata_map=adata_map, mapper=mapper, datamodule=datamodule, config=config)

    # adata_map.write(filename=f"/nfsd/sysbiobig/bazzaccoen/tangramlit_dev/results/adata_map_Dataset{dataset_number}.h5ad", )



if __name__ == "__main__":
    main()



    #TODO: create a script that takes the two adata as input and a trainin config and 
    # writes in output three anndata: the two input preprocessed and the adata_map.

    #TODO: create a script that performs an optuna tuning of lambdas and automiatically
    # trains a model to completion with that configuration.