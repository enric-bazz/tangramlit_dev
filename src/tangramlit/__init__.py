__version__ = "0.1.0"

# Import public API modules lazily to not slow down cli

_lazy_modules = {
    # mapping
    "map_cells_to_space": ".mapping.trainer",
    "run_multiple_mappings": ".mapping.trainer",
    "validate_mapping_experiment": ".mapping.utils",
    "project_sc_genes_onto_space": ".mapping.utils",
    "compare_spatial_geneexp": ".mapping.utils",
    # data
    "split_train_val_test_genes": ".data.genes_splits",
    "kfold_gene_splits": ".data.genes_splits",
    "fps_seeds": ".data.spatial_binning",
    "bfs_bin_growth": ".data.spatial_binning",
    "compute_bin_celltype_fractions": ".data.spatial_binning",
    # hpo
    "tune_loss_coefficients": ".hpo",
    # benchmarking
    "benchmark_mapping": ".benchmarking.utils", 
    "aggregate_benchmarking_metrics": ".benchmarking.utils",
    # analysis
    # Annotation
    "deterministic_annotation": ".analysis",
    "transfer_annotation": ".analysis",
    "annotation_report": ".analysis",
    "one_hot_encoding": ".analysis",
    # Cells filter
    "compute_filter_corr": ".analysis",
    "filter_cell_choice_consistency": ".analysis",
    "plot_filter_weights": ".analysis",
    "plot_filter_count": ".analysis",
    # Deterministic mapping
    "get_cell_spot_pair": ".analysis",
    "get_spot_cell_pair": ".analysis",
    "count_deterministic_mapping_matches": ".analysis",
    "deterministic_mapping_similarity": ".analysis",
    # Plotting
    "plot_training_history": ".analysis",
    "plot_loss_terms": ".analysis",
    "plot_validation_metrics_history": ".analysis",
    "plot_training_scores": ".analysis",
    "plot_auc_curve": ".analysis",
    "plot_score_SA_corr": ".analysis",
    "plot_score_histograms": ".analysis",
    # Training genes
    "traffic_light_plot": ".analysis",
    "analyze_spatial_patterns": ".analysis",
    "compute_spatial_correlations": ".analysis",
}

def __getattr__(name):
    if name in _lazy_modules:
        module = __import__(f"{__name__}{_lazy_modules[name]}", fromlist=[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")