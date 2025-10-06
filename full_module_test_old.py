# Load cropped datasets
import anndata as ad
import numpy as np

path = "C:/Users/enric/tangram/myDataCropped"
adata_sc = ad.read_h5ad(path + "/test_sc_crop.h5ad")
adata_st = ad.read_h5ad(path + "/slice200_norm_reduced.h5ad")
# considerably reduce the number of spots as refinements require to compute and store several versions of the neighbors graph

import tangramlit as tg

# Set parameters for mapping
mode = "filter"
target_count = None
cluster_label = "cluster_labels"

# Set seed for reproducibility
random_state = 123

ad_map_lt, mapper, datamodule = tg.map_cells_to_space(
    adata_sc,
    adata_st,
    mode=mode,
    target_count=target_count,
    num_epochs=200,
    lambda_d=1,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    lambda_count=1,
    lambda_f_reg=1e-5,
    lambda_l2=0,
    lambda_l1=0,
    random_state=random_state,
    lambda_sparsity_g1=1,
    lambda_neighborhood_g1=1,
    lambda_ct_islands=0,
    #cluster_label=cluster_label,
    lambda_getis_ord=1,
    lambda_moran=1,
    lambda_geary=1 ,
    )

# sc gene projection
ad_ge = tg.project_sc_genes_onto_space(ad_map_lt, datamodule)
df = tg.compare_spatial_gene_exp(ad_ge, datamodule)
tg.plot_training_scores(df)
tg.plot_auc_curve(df)

# Plot loss terms
tg.plot_loss_term(adata_map=ad_map_lt, loss_key='count_reg')
tg.plot_training_history(adata_map=ad_map_lt, hyperpams=mapper.hparams, log_scale=False, lambda_scale=True)

# Try analysis functions
df1 = tg.get_cell_spot_pair(ad_map_lt, filter=True)
df2 = tg.get_spot_cell_pair(ad_map_lt, filter=True)

filt_corr = tg.compute_filter_corr(ad_map_lt, plot=True)

# Shared labels
labels = set(adata_st.obs['class_label']) & set(adata_sc.obs['class_label'])
# Accuracy
true, pred = tg.deterministic_annotation(ad_map_lt, adata_sc, adata_st,
                            sc_cluster_label='class_label', st_cluster_label='class_label',
                            flavour='spot_to_cell', filter=False)
acc, stats = tg.annotation_report(true, pred)

map_sim = tg.deterministic_mapping_similarity(ad_map_lt, adata_sc, adata_st, flavour='spot_to_cell', filter=False)




# Plot final filter values distribution
if mode == "filter":
    tg.plot_filter_weights(ad_map_lt, plot_spaghetti=True, plot_envelope=True)
    tg.plot_filter_count(ad_map_lt, target_count=target_count)

# Annotation transfer test
tg.transfer_annotation(adata_map=ad_map_lt, adata_st=adata_st, sc_cluster_label='class_label',
                    filter=False)


### Validation
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

#Train
ad_map_lt, mapper, datamodule = tg.map_cells_to_space(
    adata_sc,
    adata_st,
    mode=mode,
    train_genes_names=train_genes,
    val_genes_names=val_genes,
    target_count=target_count,
    num_epochs=30,
    lambda_d=1,
    lambda_g1=1,
    lambda_g2=1,
    lambda_r=0.001,
    lambda_count=1,
    lambda_f_reg=1,
    lambda_l2=1e-5,
    lambda_l1=1e-5,
    random_state=random_state,
    lambda_sparsity_g1=1,
    lambda_neighborhood_g1=1,
    lambda_ct_islands=1,
    cluster_label=cluster_label,
    lambda_getis_ord=1,
    lambda_moran=1,
    lambda_geary=1 ,
    )

#Validate
results = tg.validate_mapping_experiment(mapper, datamodule)

### CROSS VALIDATION
cv_results = tg.cross_validate_mapping(adata_sc,
    adata_st,
    mode=mode,
    k=3,
input_genes=shared_genes,
    num_epochs=30,
    lambda_d=1,
    lambda_g1=1,
    lambda_g2=1,
    lambda_r=0.001,
    lambda_count=1,
    lambda_f_reg=1,
    lambda_l2=1e-5,
    lambda_l1=1e-5,
    random_state=random_state,
    lambda_sparsity_g1=1,
    lambda_neighborhood_g1=1,
    lambda_ct_islands=1,
    cluster_label=cluster_label,
    lambda_getis_ord=1,
    lambda_moran=1,
    lambda_geary=1 ,)

