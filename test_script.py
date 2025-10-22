import anndata as ad
import numpy as np
import tangramlit as tgl
import yaml


def data_loading(path):
    # Load cropped datasets
    adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
    adata_st = ad.read_h5ad(path + "/slice200_norm_reduced.h5ad")

    return (adata_sc, adata_st)

def train_mapper(data, config):
    adata_sc=data[0]
    adata_st=data[1]

    if config['filter'] and (config['target_count'] == 'None'):
        config['target_count'] = None  # remove string formatting

    adata_map, mapper, datamodule = tgl.map_cells_to_space(
        adata_sc=data[0],
        adata_st=data[1],
        **config
        )
    
    # sc gene projection
    ad_ge = tgl.project_sc_genes_onto_space(adata_map, datamodule)
    df = tgl.compare_spatial_gene_expr(ad_ge, datamodule)
    tgl.plot_training_scores(df)
    # tgl.plot_auc_curve(df)  # works only with validation genes

    # Plot loss terms
    tgl.plot_training_history(adata_map=adata_map, hyperpams=mapper.hparams, log_scale=False, lambda_scale=True)
    
    tgl.plot_loss_terms(adata_map=adata_map, loss_key=['count_reg', 'main_loss', 'ct_island_term'], lambda_coeff=[1.0e-3, 1, 1])
    tgl.plot_loss_terms(adata_map=adata_map, loss_key=['count_reg', 'main_loss', 'ct_island_term'], lambda_coeff=[1.0e-3, 1, 1],
                                                       lambda_scale=True, make_subplot=True, subplot_shape=(1,3))

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
    
    return adata_map, mapper, datamodule

def train_validate_mapper(data, config, random_state=None):
    adata_sc = data[0]
    adata_st = data[1]

    # Get shared genes (case-insensitive)
    sc_genes = {gene.lower(): gene for gene in adata_sc.var_names}
    st_genes = {gene.lower(): gene for gene in adata_st.var_names}

    # Find intersection of lowercase gene names
    shared_lower = set(sc_genes.keys()) & set(st_genes.keys())

    # Use original case from sc_genes for consistency
    shared_genes = [sc_genes[gene_lower] for gene_lower in shared_lower]

    # Random split
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle the shared genes
    shared_genes = np.array(shared_genes)
    np.random.shuffle(shared_genes)

    # Split into train and validation
    train_ratio = 0.8  # Adjust the ratio as needed
    n_train = int(len(shared_genes) * train_ratio)
    train_genes = shared_genes[:n_train]
    val_genes = shared_genes[n_train:]
    for k in ['train_genes_names', 'val_genes_names']:
        config.pop(k, None)

    #Train
    adata_map, mapper, datamodule = tgl.map_cells_to_space(
        adata_sc,
        adata_st,
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        **config
        )
    
    #Validate
    results = tgl.validate_mapping_experiment(mapper, datamodule)

    # sc gene projection
    ad_ge = tgl.project_sc_genes_onto_space(adata_map, datamodule)
    df = tgl.compare_spatial_gene_expr(ad_ge, datamodule)
    tgl.plot_training_scores(df)
    tgl.plot_auc_curve(df)

    return [adata_map, train_genes, val_genes, results], shared_genes, mapper, datamodule

def cv_mapper_genes(data, config, genes_list):
    # remove yaml input_genes
    for k in ['input_genes', 'train_genes_names', 'val_genes_names']:
        config.pop(k, None)

    cv_results = tgl.cross_validate_mapping(
        adata_sc=data[0],
        adata_st=data[1],
        input_genes=genes_list,
        **config)
    
    return cv_results


def main():

    path = "/nfsd/sysbiobig/bazzaccoen/tangramlit_dev/data"
    # path = "C:/Users/enric/tangram/myDataCropped"

    data = data_loading(path)

    with open("train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_mapper(data, config)

    # _ , shared_genes, mapper, datamodule = train_validate_mapper(data, config)

    # cv_mapper_genes(data, config, genes_list=shared_genes)
    

if __name__ == "__main__":
    main()