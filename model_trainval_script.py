import anndata as ad
import numpy as np
import tangramlit as tgl
import yaml


def load_data(data_path, spatial_tech):
    # read adata
    adata_sc = ad.read_h5ad(data_path + "scRNA_data.h5ad")
    adata_st = ad.read_h5ad(f"{data_path}{spatial_tech}_data.h5ad")

    # read config
    with open(f"{data_path}train_config.yaml", "r") as f:
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
    adata_map, _, _ = tgl.map_cells_to_space(
        adata_sc=adata_sc,
        adata_st=adata_st,
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        **config
        )
    
    return adata_map


def main():

    dataset_number = 3

    data_path = f"/nfsd/sysbiobig/bazzaccoen/tangramlit_dev/data/Dataset{dataset_number}/"

    (adata_sc, adata_st), config = load_data(data_path=data_path, spatial_tech='seqFISH')

    adata_map = train_mapper(adata_sc=adata_sc, adata_st=adata_st, config=config)

    adata_map.write(filename=f"/nfsd/sysbiobig/bazzaccoen/tangramlit_dev/results/adata_map_Dataset{dataset_number}.h5ad", )



if __name__ == "__main__":
    main()